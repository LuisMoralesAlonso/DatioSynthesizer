import dask
import distributed
import dask.dataframe as df
import pandas as pd
import numpy as np
import datiosynthesizer.config as config
import datiosynthesizer.utils as utils


def init_random(seed: int = 0) -> None:
    utils.set_random_seed(seed)


def init_generator(file_desc, n_rows: int, chunk_size: int, output_file: str, ordered_by_key: bool = False):
    #client = distributed.Client()
    init_random(config.seed)
    description = utils.read_json_file(file_desc)
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


def build_params(pos: int, name: str, describer: dict, string_length):
    params = {}
    params['type'] = describer['meta']['datatype'][name]
    params['total_chunks'] = describer['generation']['number_chunks']
    params['chunk_pos'] = pos
    params['chunk_size'] = describer['generation']['chunk_size']
    params['min'] = describer['mins'][name]
    params['min'] = describer['maxes'][name]
    params['string_length'] = np.random.randint(params['min'], params['max'])
    return params


def generate_key_chunk(params: dict):
    if params['type'] is 'SocialSecurityNumber':
        intervals = np.linspace(1, 100 - 1, num=params['total_chunks']+1, dtype=int)
        data = np.linspace(intervals[params['chunk_pos']], intervals[params['chunk_pos']+1]-1, num=params['chunk_size'], dtype=int)
        data = np.random.permutation(data)
        data = [str(i).zfill(9) for i in data]
        data = list(map(lambda i: '{}-{}-{}'.format(i[:3], i[3:5], i[5:]), data))
    elif params['type'] is 'String':
        length = np.random.randint(params['min'], params['max'])
        vectorized = np.vectorize(lambda x: '{}{}'.format(utils.generate_random_string(length), x))
        data = vectorized(np.arange(params['chunk_size']*params['cunk_pos'],params['chunk_size']*(1+params['cunk_pos'])))
        data = np.random.permutation(data)
    elif params['type'] is 'Integer':
        data = np.arange(params['min'], params['max'])
        data = np.random.permutation(data)
    elif params['type'] is 'Float' or 'Datetime':
        data = np.arange(params['min'], params['max'], (params['min'] - params['max']) / params['chunk_size'])
    else:
        data = None
    return data

def generate_string_chunk(params: dict):
    return None

def generate_int_chunk(params: dict):
    return None

def generate_float_chunk(params: dict):
    return None

def generate_datetime_chunk(params: dict):
    return None

def generate_ssn_chunk(params: dict):
    return None