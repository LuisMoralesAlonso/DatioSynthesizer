import datiosynthesizer.config as config
import datiosynthesizer.utils as utils
import dask
import copy
import numpy as np
import pandas as pd


def get_mins(datos: dict, dd: dict, describer: dict):
    # Columnas Integer/Float/Datetime/SSN las gestiona bien
    mins = {}
    # Hay que aplicar lógica a los String
    for attr in dd['meta']['attrs']:
        if describer['datatypes'][attr] is config.DataType.STRING:
            mins[attr] = dask.delayed(float)(datos['dropna'][attr].map(len, meta=('len', int)).min())
        else:
            mins[attr] = dask.delayed(float)(datos['data'][attr].min())
    return mins


def get_maxes(datos: dict, dd: dict, describer: dict):
    # Columnas Integer/Float/Datetime/SSN las gestiona bien
    maxes = {}
    # Hay que aplicar lógica a los String
    for attr in dd['meta']['attrs']:
        if describer['datatypes'][attr] is config.DataType.STRING:
            maxes[attr] = dask.delayed(float)(datos['dropna'][attr].map(len, meta=('len', int)).max())
        else:
            maxes[attr] = dask.delayed(float)(datos['data'][attr].max())
    return maxes


@dask.delayed
def get_missing_rates(datos: dict, dd: dict, describer: dict):
    rates = {}
    for attr in dd['meta']['attrs']:
        rates[attr] = float((datos['data'][attr].size - datos['dropna'][attr].size) / datos['data'][attr].size)
    return rates


@dask.delayed
def get_cat_probs(dist):
    dist_copy = copy.copy(dist)
    dist_copy.sort_index(inplace=True)
    probs = utils.normalize_given_distribution(dist_copy)
    return probs


@dask.delayed
def get_cat_bins(dist):
    dist_copy = copy.copy(dist)
    dist_copy.sort_index(inplace=True)
    bins = np.array(dist_copy.index).tolist()
    return bins

@dask.delayed
def get_noncat_probs(dist):
    dist_copy = copy.copy(dist[1])
    probs = utils.normalize_given_distribution(dist_copy)
    return probs


@dask.delayed
def get_noncat_bins(dist):
    dist_copy = copy.copy(dist[0])
    bins = dist_copy[:-1]
    bins[0] = bins[0] - 0.001 * (bins[1] - bins[0])
    bins = bins.tolist()
    return bins


def get_histograms(delayed, min, max, size):
    return [dask.delayed(np.histogram)(chunk, range=(min, max), bins=size) for chunk in delayed]


@dask.delayed
def sum_histograms(delayed):
    partials = []
    for tuple in delayed:
        partials.append(tuple[0])
    if len(partials) is 1:
        sums = partials[0]
    else:
        sums = [sum(x) for x in zip(*partials)]
    return (delayed[0][1], sums)


def get_distribution(datos: dict, dd: dict, describer: dict):
    bins = {}
    probs = {}
    for attr in dd['meta']['attrs']:
        if attr in describer['categories']:
            distribution = datos['dropna'][attr].value_counts()
            probs[attr] = get_cat_probs(distribution)
            bins[attr] = get_cat_bins(distribution)
        else:
            distribution = get_histograms(datos['dropna'][attr].to_delayed(), dd['mins'][attr], dd['maxes'][attr],
                                          config.histogram_size)
            distribution = sum_histograms(distribution)
            probs[attr] = get_noncat_probs(distribution)
            bins[attr] = get_noncat_bins(distribution)
    return probs, bins

@dask.delayed
def inject_laplace_noise(datos: dict, dd: dict):
    dists_copy = copy.copy(dd['distribution']['probs'])
    for column in dd['meta']['attrs']:
        dists_copy[column] = inject_laplace_noise_column(datos['data'][column].size,
                                                                               dd['distribution']['probs'][column],
                                                                               column, epsilon=config.epsilon,
                                                                               valid_attrs = dd['meta']['num_attrs_in_BN'])
    return dists_copy


def inject_laplace_noise_column(size, dist: dict, column, epsilon=0.1, valid_attrs=10):
    dist_copy = copy.copy(dist)
    if epsilon > 0:
        noisy_scale = valid_attrs / (epsilon * size)
        loc = 0
        scale = noisy_scale
        size = len(dist_copy)
        laplace_noises = np.random.laplace(loc, scale, size)
        noisy_distribution = np.asarray(dist_copy) + laplace_noises
        normalized = utils.normalize_given_distribution(noisy_distribution)
        return normalized


def encode_chunk_into_binning_indices(part, bn_cols: [], cat_cols: [], bin_indices: dict):
    """ Encode values into binning indices for distribution modeling."""
    part_copy = copy.copy(part)
    for col in part_copy.columns.tolist():
        if col in bn_cols:
            if col in cat_cols:
                part_copy[col] = part[col].map(utils.bin_cat(bin_indices[col]))
            else:
                part_copy[col] = part[col].map(utils.bin_noncat(bin_indices[col]))
        else:
            part_copy.drop(col, axis=1)
    return part_copy
