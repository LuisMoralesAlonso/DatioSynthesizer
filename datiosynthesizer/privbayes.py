import random
import warnings
from itertools import combinations, product
from math import log, ceil
import dask
import dask.dataframe as df
from scipy import sparse as sp
from math import log
from collections import Counter
import functools
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from datiosynthesizer.utils import normalize_given_distribution


def sensitivity(num_tuples):
    """Sensitivity function for Bayesian network construction. PrivBayes Lemma 1.

    Parameters
    ----------
        num_tuples : int
            Number of tuples in sensitive dataset.

    Return
    --------
    int
        Sensitivity value.
    """
    a = (2 / num_tuples) * log((num_tuples + 1) / 2)
    b = (1 - 1 / num_tuples) * log(1 + 2 / (num_tuples - 1))
    return a + b


def delta(num_attributes, num_tuples, epsilon):
    """Computing delta, which is a factor when applying differential privacy.

    More info is in PrivBayes Section 4.2 "A First-Cut Solution".

    Parameters
    ----------
        num_attributes : int
            Number of attributes in dataset.
        num_tuples : int
            Number of tuples in dataset.
        epsilon : float
            Parameter of differential privacy.
    """
    return 2 * (num_attributes - 1) * sensitivity(num_tuples) / epsilon


def usefulness_minus_target(k, num_attributes, num_tuples, target_usefulness=5, epsilon=0.1):
    """Usefulness function in PrivBayes.

    Parameters
    ----------
        k : int
            Max number of degree in Bayesian networks construction
        num_attributes : int
            Number of attributes in dataset.
        num_tuples : int
            Number of tuples in dataset.
        target_usefulness : int or float
        epsilon : float
            Parameter of differential privacy.
    """
    if k == num_attributes:
        print('here')
        usefulness = target_usefulness
    else:
        usefulness = num_tuples * epsilon / ((num_attributes - k) * (2 ** (k + 3)))  # PrivBayes Lemma 3
    return usefulness - target_usefulness


def calculate_k(num_attributes, num_tuples, target_usefulness=4, epsilon=0.1):
    """Calculate the maximum degree when constructing Bayesian networks. See PrivBayes Lemma 3."""
    default_k = 3
    initial_usefulness = usefulness_minus_target(default_k, num_attributes, num_tuples, 0, epsilon)
    if initial_usefulness > target_usefulness:
        return default_k
    else:
        arguments = (num_attributes, num_tuples, target_usefulness, epsilon)
        warnings.filterwarnings("error")
        try:
            ans = fsolve(usefulness_minus_target, int(num_attributes / 2), args=arguments)[0]
            ans = ceil(ans)
        except RuntimeWarning:
            print("Warning: k is not properly computed!")
            ans = default_k
        if ans < 1 or ans > num_attributes:
            ans = default_k
        return ans


def greedy_bayes(dataset, k=0, epsilon=0):
    """Construct a Bayesian Network (BN) using greedy algorithm.

    Parameters
    ----------
        dataset : DataFrame
            Input dataset, which only contains categorical attributes.
        k : int
            Maximum degree of the constructed BN. If k=0, k is automatically calculated.
        epsilon : float
            Parameter of differential privacy.
    """

    num_tuples, num_attributes = dataset.shape
    if not k:
        k = calculate_k(num_attributes, num_tuples)

    attributes = set(dataset.keys())
    N = []
    V = set()
    V.add(random.choice(attributes))

    print('================== Constructing Bayesian Network ==================')
    for i in range(1, len(attributes)):
        print('Looking for next attribute-parents pair.')
        rest_attributes = attributes - V
        parents_pair_list = []
        mutual_info_list = []
        for child in rest_attributes:
            print('    Considering attribute {}'.format(child))
            for parents in combinations(V, min(k, len(V))):
                parents = list(parents)
                parents_pair_list.append((child, parents))
                # TODO consider to change the computation of MI by combined integers instead of strings.
                mi = mutual_information(dataset[child], dataset[parents])
                mutual_info_list.append(mi)

        if epsilon:
            sampling_distribution = exponential_mechanism(dataset, mutual_info_list, epsilon)
            idx = np.random.choice(list(range(len(mutual_info_list))), p=sampling_distribution)
        else:
            idx = mutual_info_list.index(max(mutual_info_list))

        N.append(parents_pair_list[idx])
        V.add(parents_pair_list[idx][0])

    print('========================= BN constructed =========================')

    return N


def exponential_mechanism(dataset, mutual_info_list, epsilon=0.1):
    """Applied in Exponential Mechanism to sample outcomes."""
    num_tuples, num_attributes = dataset.shape
    mi_array = np.array(mutual_info_list)
    mi_array = mi_array / (2 * delta(num_attributes, num_tuples, epsilon))
    mi_array = np.exp(mi_array)
    mi_array = normalize_given_distribution(mi_array)
    return mi_array


def laplacian_noise_parameter(k, num_attributes, num_tuples, epsilon):
    """The noises injected into conditional distributions. PrivBayes Algorithm 1."""
    return 4 * (num_attributes - k) / (num_tuples * epsilon)


def get_noisy_distribution_of_attributes(attributes, encoded_dataset, epsilon=0.1):
    data = encoded_dataset.copy().loc[:, attributes]
    data['count'] = 1
    stats = data.groupby(attributes).sum()

    full_space = pd.DataFrame(columns=attributes, data=list(product(*stats.index.levels)))
    stats.reset_index(inplace=True)
    stats = pd.merge(full_space, stats, how='left')
    stats.fillna(0, inplace=True)

    if epsilon:
        k = len(attributes) - 1
        num_tuples, num_attributes = encoded_dataset.shape
        noise_para = laplacian_noise_parameter(k, num_attributes, num_tuples, epsilon)
        laplacian_noises = np.random.laplace(0, scale=noise_para, size=stats.index.size)
        stats['count'] += laplacian_noises
        stats.loc[stats['count'] < 0, 'count'] = 0

    return stats


def construct_noisy_conditional_distributions(bayesian_network, encoded_dataset, epsilon=0.1):
    """See more in Algorithm 1 in PrivBayes."""

    k = len(bayesian_network[-1][1])
    conditional_distributions = {}

    # first k+1 attributes
    root = bayesian_network[0][1][0]
    kplus1_attributes = [root]
    for child, _ in bayesian_network[:k]:
        kplus1_attributes.append(child)

    noisy_dist_of_kplus1_attributes = get_noisy_distribution_of_attributes(kplus1_attributes, encoded_dataset,
                                                                           epsilon)

    # generate noisy distribution of root attribute.
    root_stats = noisy_dist_of_kplus1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
    conditional_distributions[root] = normalize_given_distribution(root_stats).tolist()

    for idx, (child, parents) in enumerate(bayesian_network):
        conditional_distributions[child] = {}

        if idx < k:
            stats = noisy_dist_of_kplus1_attributes.copy().loc[:, parents + [child, 'count']]
        else:
            stats = get_noisy_distribution_of_attributes(parents + [child], encoded_dataset, epsilon)

        stats = pd.DataFrame(stats.loc[:, parents + [child, 'count']].groupby(parents + [child]).sum())

        if len(parents) == 1:
            for parent_instance in stats.index.levels[0]:
                dist = normalize_given_distribution(stats.loc[parent_instance]['count']).tolist()
                conditional_distributions[child][str([parent_instance])] = dist
        else:
            for parents_instance in product(*stats.index.levels[:-1]):
                dist = normalize_given_distribution(stats.loc[parents_instance]['count']).tolist()
                conditional_distributions[child][str(list(parents_instance))] = dist

    return conditional_distributions

#Both contingency matrix and MI for each partition
@dask.delayed
def partition_mutual_info_pre_score(true: pd.Series, pred: pd.Series):
    datos = {}
    true_classes, true_idx = np.unique(true, return_inverse=True)
    datos['true_classes'] = true_classes
    datos['true_idx'] = true_idx
    pred_classes, pred_idx = np.unique(pred, return_inverse=True)
    datos['pred_classes'] = pred_classes
    datos['pred_idx'] = pred_idx
    n_classes = true_classes.shape[0]
    n_preds = pred_classes.shape[0]
    datos['n_classes'] = n_classes
    datos['n_preds'] = n_preds
    contingency = sp.coo_matrix((np.ones(true_idx.shape[0]),
                                 (true_idx, pred_idx)),
                                shape=(n_classes, n_preds),
                                dtype=np.int)
    nzx, nzy, nz_val = sp.find(contingency)
    datos['nzx'], datos['nzy'], datos['nz_val'] = nzx, nzy, nz_val
    contingency_sum = contingency.sum()
    datos['contingency_sum'] = contingency_sum
    pi = np.ravel(contingency.sum(axis=1))
    datos['pi'] = pi
    pj = np.ravel(contingency.sum(axis=0))
    datos['pj'] = pj
    return datos


@dask.delayed(nout=2)
def gen_pi_pj(chunks_mi_list: list, true_classes: list, pred_classes: list):
    #pi_dask = [0 for i in range(true_classes_len)]
    pi_dask = np.zeros(len(true_classes))
    pj_dask = np.zeros(len(pred_classes))
    for mi_chunk in chunks_mi_list:
        for index, clase in enumerate(true_classes):
            try:
                index_true_clase = mi_chunk['true_classes'].tolist().index(clase)
                pi_dask[index] = pi_dask[index] + mi_chunk['pi'][mi_chunk['true_classes'].tolist().index(clase)]
            except (IndexError, ValueError):
                None
        for index, clase in enumerate(pred_classes):
            try:
                index_pred_clase = mi_chunk['pred_classes'].tolist().index(clase)
                pj_dask[index] = pj_dask[index] + mi_chunk['pj'][mi_chunk['pred_classes'].tolist().index(clase)]
            except (IndexError, ValueError):
                None
    return (pi_dask, pj_dask)


@dask.delayed(nout=3)
def gen_nzx_nzy_nzval_dask(chunks_mi_list: list, true_classes, pred_classes):
    nzx_dask, nzy_dask, nz_val_dask = np.array([], dtype=np.int64),np.array([], dtype=np.int64),np.array([], dtype=np.int64)
    cross_clusters_list = []
    for mi_chunk in chunks_mi_list:
        true_nzx_np = np.array(list(map(lambda x: mi_chunk['true_classes'][x], mi_chunk['nzx'])))
        true_nzy_np = np.array(list(map(lambda x: mi_chunk['pred_classes'][x], mi_chunk['nzy'])))
        true_nz_val = mi_chunk['nz_val']
        cross_clusters_list.append(Counter(dict(list(zip(zip(true_nzx_np,true_nzy_np),true_nz_val)))))
    cross_clusters = dict(functools.reduce(lambda a,b : a+b,cross_clusters_list))
    for key in cross_clusters.keys():
        nzx_dask = np.append(nzx_dask, true_classes.tolist().index(key[0]))
        nzy_dask = np.append(nzy_dask, pred_classes.tolist().index(key[1]))
        nz_val_dask = np.append(nz_val_dask, cross_clusters[key])
    return (nzx_dask, nzy_dask, nz_val_dask)


@dask.delayed
def get_log_outer(outer_delayed, pi_delayed, pj_delayed):
    print(outer_delayed)
    print(pi_delayed)
    print(pj_delayed)
    return -np.log(outer_delayed) + np.log(sum(pi_delayed)) + np.log(sum(pj_delayed))

@dask.delayed
def get_mi(contingency_nm_d, log_contingency_nm_d, contingency_sum_d, log_outer_d):
    return (contingency_nm_d * (log_contingency_nm_d - log(contingency_sum_d)) +
          contingency_nm_d * log_outer_d)

@dask.delayed
def get_contingency_sum(chunks_mi_list: list):
    suma = 0
    for mi_chunk in chunks_mi_list:
        suma = suma + mi_chunk['contingency_sum']
    print(suma)
    return suma

@dask.delayed
def get_log_contingency_nm(nz_val_delayed):
    return np.log(nz_val_delayed)

@dask.delayed
def get_contingency_nm(contingency_sum_delayed, nz_val_delayed):
    contingency_nm = nz_val_delayed / contingency_sum_delayed
    return contingency_nm

def mutual_information(true: df.DataFrame, pred: df.DataFrame):
    # Mutual information of distributions in format of pd.Series or pd.DataFrame.
    str_trues = true.astype(str).apply(lambda x: ' '.join(x.tolist()), axis=1, meta=('phrase', 'object'))
    str_preds = pred.astype(str).apply(lambda x: ' '.join(x.tolist()), axis=1, meta=('phrase', 'object'))
    true_classes = str_trues.unique()
    pred_classes = str_preds.unique()
    chunked_mi_list = list(map(lambda x: partition_mutual_info_pre_score(x[0], x[1]), list(zip(str_trues.to_delayed(),
                                                                                               str_preds.to_delayed()))))
    pi, pj = gen_pi_pj(chunked_mi_list, true_classes, pred_classes)
    nzx, nzy, nz_val = gen_nzx_nzy_nzval_dask(chunked_mi_list, true_classes, pred_classes)
    contingency_sum = get_contingency_sum(chunked_mi_list)
    log_contingency_nm = get_log_contingency_nm(nz_val)
    contingency_nm = get_contingency_nm(contingency_sum, nz_val)

    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64) * pj.take(nzy).astype(np.int64)
    log_outer = get_log_outer(outer, pi, pj)
    mi = get_mi(contingency_nm, log_contingency_nm, contingency_sum, log_outer)
    return mi
