import dask
import distributed
import dask.dataframe as df
import pandas as pd
import numpy as np
import datiosynthesizer.config as config
import datiosynthesizer.utils as utils
import time, datetime


def init_random(seed: int = 0) -> None:
    utils.set_random_seed(seed)


def init_generator(file_desc, n_rows: int, chunk_size: int, output_file: str, ordered_by_key: bool = False):
    #client = distributed.Client()
    init_random(config.seed)
    description = utils.read_json_file(file_desc)
    description['generation'] = {}
    description['generation']['n_rows'] = n_rows
    description['generation']['output_file'] = output_file
    description['generation']['ordered_by_key'] = ordered_by_key
    description['generation']['chunk_size'] = chunk_size
    rest_rows = n_rows % chunk_size
    description['generation']['number_chunks'] = int(n_rows // chunk_size)
    if rest_rows:
        description['generation']['last_chunk_size'] = rest_rows
    else:
        description['generation']['last_chunk_size'] = 0
    return description


def generate_random(desc: dict):
    delayed_chunks = []
    global_conf = build_conf(desc)
    for chunk_pos in range(desc['generation']['number_chunks']):
        delayed_chunks.append(write_chunk('random', desc, global_conf, chunk_pos, desc['generation']['output_file']))
    delayed_chunks = dask.compute(*delayed_chunks)
    return delayed_chunks


def generate_independent(desc: dict):
    delayed_chunks = []
    global_conf = build_conf(desc)
    for chunk_pos in range(desc['generation']['number_chunks']):
        delayed_chunks.append(write_chunk('independent', desc, global_conf, chunk_pos, desc['generation']['output_file']))
    delayed_chunks = dask.compute(*delayed_chunks)
    return delayed_chunks


def build_conf(desc: dict):
    dataset_params = {}
    for column in desc['meta']['attrs']:
        dataset_params[column] = build_params(column, desc)
    return dataset_params

def build_params(name: str, describer: dict):
    params = {}
    params['type'] = describer['meta']['datatypes'][name]
    params['key'] = False
    if name in describer['meta']['key_candidates']:
        params['key'] = True
    params['categorical'] = False
    if name in describer['meta']['categories']:
        params['categorical'] = True
    params['total_chunks'] = describer['generation']['number_chunks']
    params['chunk_size'] = describer['generation']['chunk_size']
    params['min'] = describer['mins'][name]
    params['max'] = describer['maxes'][name]
    params['string_length'] = np.random.randint(params['min'], params['max'])
    params['distribution_bins'] = describer['distribution']['bins'][name]
    return params

@dask.delayed
def write_chunk(gen_type: str, description: dict, conf: dict, pos: int, output_file: str):
    start_time = time.time()
    now = datetime.datetime.now()
    print('Init ' + str(pos) + ': ' + now.strftime("%Y-%m-%d %H:%M:%S:%f"))
    if gen_type == 'random':
        chunk = generate_rand_chunk(description, conf, pos)
    elif gen_type == 'independent':
        chunk = generate_ind_chunk(description, conf, pos)
    elapsed_time = time.time() - start_time
    print('chunk ' + str(pos) + ': ' + str(elapsed_time))
    output = output_file + str(pos) + str('.parquet')
    start_time = time.time()
    chunk.to_parquet(output, engine='fastparquet')
    elapsed_time = time.time() - start_time
    print('write ' + str(pos) + ': ' + str(elapsed_time))
    return output


def generate_rand_chunk(description: dict, conf: dict, pos: int):
    length = conf[list(conf.keys())[0]]['chunk_size']
    data_chunk = pd.DataFrame(index=range(pos * length, (pos + 1) * length))
    for key in conf.keys():
        params = conf[key]
        params['chunk_pos'] = pos
        if params['key']:
            data_chunk[key] = generate_key_chunk(params)
        elif params['categorical']:
            data_chunk[key] = generate_rand_cat_chunk(params)
        elif params['type'] == 'String':
            data_chunk[key] = generate_rand_string_chunk(params)
        else:
            if params['type'] == 'Integer':
                data_chunk[key] = generate_rand_int_chunk(params)
            else:
                data_chunk[key] = generate_rand_int_chunk(params)
    return data_chunk


def generate_ind_chunk(description: dict, conf: dict, pos: int):
    length = conf[list(conf.keys())[0]]['chunk_size']
    data_chunk = pd.DataFrame(index=range(pos * length, (pos + 1) * length))
    for key in conf.keys():
        params = conf[key]
        params['chunk_pos'] = pos
        if params['key']:
            data_chunk[key] = generate_key_chunk(params)
        elif params['categorical']:
            data_chunk[key] = generate_ind_cat_chunk(params)
        elif params['type'] == 'String':
            data_chunk[key] = generate_ind_string_chunk(params)
        else:
            if params['type'] == 'Integer':
                data_chunk[key] = generate_ind_int_chunk(params)
            else:
                data_chunk[key] = generate_ind_float_datetime_chunk(params)
    return data_chunk


def generate_key_chunk(params: dict):
    if params['type'] == 'SocialSecurityNumber':
        intervals = np.linspace(1, 100 - 1, num=params['total_chunks']+1, dtype=int)
        data = np.linspace(intervals[params['chunk_pos']], intervals[params['chunk_pos']+1]-1, num=params['chunk_size'], dtype=int)
        data = np.random.permutation(data)
        data = [str(i).zfill(9) for i in data]
        data = list(map(lambda i: '{}-{}-{}'.format(i[:3], i[3:5], i[5:]), data))
    elif params['type'] == 'String':
        length = np.random.randint(params['min'], params['max'])
        vectorized = np.vectorize(lambda x: '{}{}'.format(utils.generate_random_string(length), x))
        data = vectorized(np.arange(params['chunk_size']*params['chunk_pos'],params['chunk_size']*(1+params['chunk_pos'])))
        data = np.random.permutation(data)
    elif params['type'] == 'Integer' or 'Datetime':
        intervals = np.linspace(params['min'], params['max'], num=params['total_chunks'] + 1, dtype=int)
        data = np.random.randint(intervals[params['chunk_pos']], intervals[params['chunk_pos']+1]-1, params['chunk_size'])
        data = np.random.permutation(data)
    elif params['type'] == 'Float':
        intervals = np.linspace(params['min'], params['max'], num=params['total_chunks'] + 1, dtype=int)
        range = intervals[params['chunk_pos'] + 1] - 1 - intervals[params['chunk_pos']]
        data = intervals[params['chunk_pos']] + np.random.sample(params['chunk_size']) * range
        data = np.random.permutation(data)
    else:
        data = None
    return data

def generate_rand_cat_chunk(params: dict):
    data = np.random.choice(params['distribution_bins'], params['chunk_size'])
    return data

def generate_rand_int_chunk(params: dict):
    return np.random.randint(params['min'], params['max'] + 1, params['chunk_size'])

def generate_rand_float_datetime_chunk(params: dict):
    return np.random.uniform(params['min'], params['max'], params['chunk_size'])

def generate_rand_string_chunk(params: dict):
    length = np.random.randint(params['min'], params['max'])
    vectorized = np.vectorize(lambda x: utils.generate_random_string(length))
    data = vectorized(np.arange(params['chunk_size']))
    return data

def generate_ind_cat_chunk(params: dict):
    data = np.random.choice(params['distribution_bins'], params['chunk_size'])
    return data

def generate_ind_int_chunk(params: dict):
    return np.random.randint(params['min'], params['max'] + 1, params['chunk_size'])

def generate_ind_float_datetime_chunk(params: dict):
    return np.random.uniform(params['min'], params['max'], params['chunk_size'])

def generate_ind_string_chunk(params: dict):
    length = np.random.randint(params['min'], params['max'])
    vectorized = np.vectorize(lambda x: utils.generate_random_string(length))
    data = vectorized(np.arange(params['chunk_size']))
    return data