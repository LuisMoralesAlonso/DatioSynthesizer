import dask
import dask.dataframe as df
import numpy as np
import pandas as pd
from scipy import sparse as sp
from math import log
from collections import Counter
import functools
import sklearn.metrics as metrics
np.random.seed(0)
import datiosynthesizer.privbayes as privbayes

trues = df.from_pandas(pd.DataFrame([1,2,3,3,2,1]), npartitions=2)
preds = df.from_pandas(pd.DataFrame([4,5,4,4,5,5]), npartitions=2)
mi = privbayes.mutual_information(trues, preds)

mi = dask.compute(mi)[0].sum()

print(mi)