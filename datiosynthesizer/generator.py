import dask
import distributed
import pandas as pd
import numpy as np
import datiosynthesizer.config as config
import datiosynthesizer.utils as utils
import time, datetime
from numpy.random import choice
from random import uniform


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
    description['generation']['number_chunks'] = int(n_rows // chunk_size)
    rest_rows = n_rows % chunk_size
    if rest_rows:
        description['generation']['last_chunk_size'] = rest_rows
        description['generation']['number_chunks']+=1
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


def generate_correlated(desc: dict):
    delayed_chunks = []
    global_conf = build_conf(desc)
    for chunk_pos in range(desc['generation']['number_chunks']):
        delayed_chunks.append(write_chunk('correlated', desc, global_conf, chunk_pos, desc['generation']['output_file']))
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
    params['last_chunk_size'] = describer['generation']['last_chunk_size']
    params['min'] = describer['mins'][name]
    params['max'] = describer['maxes'][name]
    params['string_length'] = np.random.randint(params['min'], params['max'])
    params['distribution_bins'] = describer['distribution']['bins'][name]
    params['distribution_probs'] = describer['distribution']['probs'][name]
    return params

#@dask.delayed
def write_chunk(gen_type: str, description: dict, conf: dict, pos: int, output_file: str):
    start_time = time.time()
    now = datetime.datetime.now()
    print('Init ' + str(pos) + ': ' + now.strftime("%Y-%m-%d %H:%M:%S:%f"))
    if gen_type == 'random':
        chunk = generate_rand_chunk(conf, pos)
    elif gen_type == 'independent':
        chunk = generate_ind_chunk(conf, pos)
    elif gen_type == 'correlated':
        chunk = generate_correl_chunk(description, conf, pos)
    else: None
    elapsed_time = time.time() - start_time
    print('chunk ' + str(pos) + ': ' + str(elapsed_time))
    output = output_file + str(pos) + str('.parquet')
    start_time = time.time()
    chunk.to_parquet(output, engine='fastparquet')
    elapsed_time = time.time() - start_time
    print('write ' + str(pos) + ': ' + str(elapsed_time))
    return output


def generate_rand_chunk(conf: dict, pos: int):
    if (pos+1 == conf[list(conf.keys())[0]]['total_chunks']) and conf[list(conf.keys())[0]]['last_chunk_size']:
        for column in conf.keys():
            conf[column]['chunk_size'] = conf[column]['last_chunk_size']
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


def generate_ind_chunk(conf: dict, pos: int):
    if (pos + 1 == conf[list(conf.keys())[0]]['total_chunks']) and conf[list(conf.keys())[0]]['last_chunk_size']:
        for column in conf.keys():
            conf[column]['chunk_size'] = conf[column]['last_chunk_size']
    length = conf[list(conf.keys())[0]]['chunk_size']
    data_chunk = pd.DataFrame(index=range(pos * length, (pos + 1) * length))
    for key in conf.keys():
        params = conf[key]
        params['chunk_pos'] = pos
        if params['key']:
            data_chunk[key] = generate_key_chunk(params)
        else:
            binning_indices = pd.Series(choice(len(params['distribution_probs']), size=length, p=params['distribution_probs']))
            data = binning_indices.apply(lambda x: uniform_sampling_within_a_bin(params, x))
            if params['type'] == 'SocialSecurityNumber' or 'Float':
                data_chunk[key] = data
            elif params['type'] == 'String':
                data_chunk[key] = generate_ind_string_chunk(params, data)
            elif params['type'] == 'Integer' or 'Datetime':
                data_chunk[key] = generate_ind_int_datetime_chunk(data)
            else:
                None
    return data_chunk


def generate_correl_chunk(desc: dict, conf: dict, pos: int):
    if (pos + 1 == conf[list(conf.keys())[0]]['total_chunks']) and conf[list(conf.keys())[0]]['last_chunk_size']:
        for column in conf.keys():
            conf[column]['chunk_size'] = conf[column]['last_chunk_size']
    length = conf[list(conf.keys())[0]]['chunk_size']
    data_chunk = pd.DataFrame(index=range(pos * length, (pos + 1) * length))
    data_encoded = generate_encoded_chunk(length, desc)
    for key in conf.keys():
        params = conf[key]
        params['chunk_pos'] = pos
        if key in data_encoded.keys():
            data = data_encoded[key].apply(lambda x: uniform_sampling_within_a_bin(params, x))
            if params['type'] == 'SocialSecurityNumber' or 'Float':
                data_chunk[key] = data
            elif params['type'] == 'String':
                data_chunk[key] = generate_ind_string_chunk(params, data)
            elif params['type'] == 'Integer' or 'Datetime':
                data_chunk[key] = generate_ind_int_datetime_chunk(data)
            else:
                None
        elif params['key']:
            data_chunk[key] = generate_key_chunk(params)
        else:
            binning_indices = pd.Series(choice(len(params['distribution_probs']), size=length, p=params['distribution_probs']))
            data = binning_indices.apply(lambda x: uniform_sampling_within_a_bin(params, x))
            if params['type'] == 'SocialSecurityNumber' or 'Float':
                data_chunk[key] = data
            elif params['type'] == 'String':
                data_chunk[key] = generate_ind_string_chunk(params, data)
            elif params['type'] == 'Integer' or 'Datetime':
                data_chunk[key] = generate_ind_int_datetime_chunk(data)
            else:
                None
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


def generate_ind_int_datetime_chunk(column):
    column[~column.isnull()] = column[~column.isnull()].astype(int)
    return column

def generate_ind_string_chunk(params: dict, column):
    if not params['categorical']:
        column[~column.isnull()] = column[~column.isnull()].apply(lambda x: utils.generate_random_string(int(x)))
    return column


def uniform_sampling_within_a_bin(params, binning_index):
    binning_index = int(binning_index)
    if binning_index == len(params['distribution_bins']):
        return np.nan
    elif params['categorical']:
        return params['distribution_bins'][binning_index]
    else:
        bins = params['distribution_bins'].copy()
        bins.append(2 * bins[-1] - bins[-2])
        return uniform(bins[binning_index], bins[binning_index + 1])


def generate_encoded_chunk(n, description):
    bn = description['bayesian_network']
    bn_root_attr = bn[0][1][0]
    root_attr_dist = description['conditional_probabilities'][bn_root_attr]
    encoded_df = pd.DataFrame(columns=utils.get_sampling_order(bn))
    encoded_df[bn_root_attr] = np.random.choice(len(root_attr_dist), size=n, p=root_attr_dist)

    for child, parents in bn:
        child_conditional_distributions = description['conditional_probabilities'][child]
        for parents_instance in child_conditional_distributions.keys():
            dist = child_conditional_distributions[parents_instance]
            parents_instance = list(eval(parents_instance))

            filter_condition = ''
            for parent, value in zip(parents, parents_instance):
                filter_condition += '({0}["{1}"]=={2}) & '.format('encoded_df', parent, value)

            filter_condition = eval(filter_condition[:-3])

            size = encoded_df[filter_condition].shape[0]
            if size:
                encoded_df.loc[filter_condition, child] = np.random.choice(len(dist), size=size, p=dist)

        unconditioned_distribution = description['attribute_description'][child]['distribution_probabilities']
        encoded_df.loc[encoded_df[child].isnull(), child] = np.random.choice(len(unconditioned_distribution),
                                                                             size=encoded_df[child].isnull().sum(),
                                                                             p=unconditioned_distribution)
    encoded_df[encoded_df.columns] = encoded_df[encoded_df.columns].astype(int)
    return encoded_df