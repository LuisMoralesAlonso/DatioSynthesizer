import json
import random
import math
import bisect
from string import ascii_lowercase
import dask
import dask.dataframe as df
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


def set_random_seed(seed: object = 0) -> object:
    random.seed(seed)
    np.random.seed(seed)


def pairwise_attributes_mutual_information(dataset):
    """Compute normalized mutual information for all pairwise attributes. Return a DataFrame."""
    mi_df = pd.DataFrame(columns=dataset.columns, index=dataset.columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str))
    return mi_df


def normalize_given_distribution(frequencies):
    distribution = np.array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    if distribution.sum() == 0:
        distribution.fill(1 / distribution.size)
    else:
        distribution = distribution / distribution.sum()
    return distribution.tolist()


def read_json_file(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)


def infer_numerical_attributes_in_dataframe(dataframe):
    describe = dataframe.describe()
    # pd.DataFrame.describe() usually returns 8 rows.
    if describe.shape[0] == 8:
        return describe.columns.tolist()
    # pd.DataFrame.describe() returns less than 8 rows when there is no numerical attribute.
    else:
        return []


def isnan(value):
    try:
        return math.isnan(float(value))
    except:
        return False


def bin_cat(x, bins):
    if isnan(x):
        return len(bins)
    else:
        return bins.index(x)


def bin_noncat(x, bins):
    if isnan(x):
        return len(bins)
    else:
        return bisect.bisect_left(bins, x) - 1


def display_bayesian_network(bn):
    length = 0
    for child, _ in bn:
        if len(child) > length:
            length = len(child)

    print('Constructed Bayesian network:')
    for child, parents in bn:
        print("    {0:{width}} has parents {1}.".format(child, parents, width=length))


def generate_random_string(length):
    return ''.join(np.random.choice(list(ascii_lowercase), size=length))
