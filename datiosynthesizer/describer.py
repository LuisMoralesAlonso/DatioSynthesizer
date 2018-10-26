import dask
import dask.dataframe as df
import distributed
import numpy as np
import json
import datiosynthesizer.config as config
import datiosynthesizer.utils as utils
import datiosynthesizer.attributes as attributes
from datiosynthesizer.privbayes import greedy_bayes, construct_noisy_conditional_distributions


describer = {}


def init_random(seed: int = 0) -> None:
    utils.set_random_seed(seed)


def init_describer(seed: int = 0, epsilon: float = 0.1, attribute_to_datatype: dict = {},
                     attribute_to_is_categorical: list = [], attribute_to_is_candidate_key: list = []) -> None:
    #client = distributed.Client()
    describer['datatypes'] = {attr: config.DataType(data_type) for attr, data_type in
                                              attribute_to_datatype.items()}
    describer['categories'] = attribute_to_is_categorical
    describer['key_candidates'] = attribute_to_is_candidate_key
    init_random(config.seed)


def independent_mode(dataset_file: object, alone=True) -> dict:
    dd = {}
    datos= {}
    datos['data'] = read_csv(dataset_file)
    datos['dropna'] = dropna(datos['data'])
    dd['meta'] = init_meta(datos, describer)
    dd['meta']['datatypes'] = describer['datatypes']
    dd['meta']['categories'] = describer['categories']
    dd['meta']['key_candidates'] = describer['key_candidates']
    dd['mins'] = attributes.get_mins(datos, dd, describer)
    dd['maxes'] = attributes.get_maxes(datos, dd, describer)
    dd['missing_rates'] = attributes.get_missing_rates(datos, dd, describer)
    dd['distribution'] = {}
    dd['distribution']['probs'],dd['distribution']['bins'] = attributes.get_distribution(datos, dd, describer)
    #First noise injection for differential privacy
    dd['distribution']['probs'] = attributes.inject_laplace_noise(datos, dd)
    if alone:
        dd = dask.optimize(dd)[0]
        dd = dask.compute(dd)[0]
    return dd, datos

def random_mode(dataset_file: object) -> dict:
    dd, data = independent_mode(dataset_file, alone=False)
    # After running independent attribute mode, 1) make all distributions uniform; 2) set missing rate to zero.
    for attr in dd['meta']['attrs']:
        distribution = dd['distribution']['probs'][attr]
        uniform_distribution = np.ones_like(distribution)
        uniform_distribution = utils.normalize_given_distribution(uniform_distribution).tolist()
        dd['distribution']['probs'][attr] = uniform_distribution
        dd['missing_rate'][attr] = 0
    dd = dask.optimize(dd)[0]
    dd = dask.compute(dd)[0]
    return dd, data


def correlated_mode(dataset_file: object) -> dict:
    dd, data = independent_mode(dataset_file, alone=True)
    data['encoded_dataset'] = df.from_delayed(encode_dataset_into_binning_indices(dd, data['data'], dd['meta']['attrs_in_BN'],
                                                                                  describer['categories']))
    #if data['encoded_dataset'].shape[1] < 2:
    #    raise Exception("Constructing Bayesian Network needs more attributes.")

    bayesian_network = greedy_bayes(data['encoded_dataset'], dd, config.k, config.epsilon)
    dd['bayesian_network'] = bayesian_network
    #bayesian_network=[('marital-status', ['relationship']), ('sex', ['relationship', 'marital-status']),
    #                        ('education', ['relationship', 'sex']), ('age', ['education', 'marital-status']),
    #                        ('income', ['education', 'sex'])]
    dd['conditional_probabilities'] = construct_noisy_conditional_distributions(
        bayesian_network, data['encoded_dataset'], config.k, config.epsilon)
    return dd, data


def dropna(data: df):
    dropna = {}
    for attr in data.columns.tolist():
        dropna[attr] = data[attr].dropna()
    return dropna

#TODO: Add support for S3, HDFS, etc and other formats like parquet
def read_csv(dataset_file: object) -> df.DataFrame:
    data = df.read_csv(dataset_file, skipinitialspace=True, na_values=config.null_values)
    data['ssn'] = data['ssn'].map(lambda x: int(x.replace('-', '')),meta=('ssn',int))
    #TODO: Procesar las columnas declaradas como fechas
    return data


def init_meta(datos: dict, describer: dict):
    columns = list(describer['datatypes'].keys())
    non_categorical_string_attributes = [attr for attr in columns if
                                         (attr not in describer['categories'])
                                         and (describer['datatypes'][attr] is config.DataType.STRING)]
    attributes_in_bn = list(set(columns) - set(describer['key_candidates']) - set(non_categorical_string_attributes))
    num_attributes_in_bn = len(attributes_in_bn)
    return {"num_tuples": datos['data'].shape[0].compute(),
                                   "num_attrs": datos['data'].shape[1],
                                   "num_attrs_in_BN": num_attributes_in_bn,
                                   "attrs": columns,
                                   "key_candidates": describer['key_candidates'],
                                   "non_cat_string": non_categorical_string_attributes,
                                   "attrs_in_BN": attributes_in_bn}


def encode_dataset_into_binning_indices(dd: dict, data: df, bn_attrs: [], cat_attrs: []):
    """Before constructing Bayesian network, encode input dataset into binning indices."""
    data_enconded = data.to_delayed()
    data_enconded = [dask.delayed(attributes.encode_chunk_into_binning_indices)
                     (chunk, bn_attrs, cat_attrs, dd['distribution']['bins']) for chunk in data_enconded]
    return data_enconded

def save_dataset_description_to_file(description: dict, file_name):
    with open(file_name, 'w') as outfile:
        json.dump(description, outfile, indent=4)














