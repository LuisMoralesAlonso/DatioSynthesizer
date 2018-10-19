import dask
import dask.dataframe as df
import numpy as np
import pandas as pd
np.random.seed(0)
import datiosynthesizer.privbayes as privbayes

dataset = df.from_pandas(pd.DataFrame(data={'col1':[1,2,3,3,2,1], 'col2':[4,5,4,4,5,5], 'col3':[3,4,3,4,3,4],
                                            'col4':[1,2,3,4,5,6], 'col5':[3,3,3,3,3,3]}), npartitions=2)
print(dataset)
describer = {}
describer['num_tuples'] = 6
describer['num_attrs_in_BN'] = 5
describer['attrs_in_BN'] = ['col1', 'col2', 'col3', 'col4', 'col5']

net=privbayes.greedy_bayes(dataset, describer, k=2, epsilon=0.1)
print(net)
#Prueba unitaria de mutual_information
comm='''
trues = df.from_pandas(pd.DataFrame([1,2,3,3,2,1]), npartitions=2)
preds = df.from_pandas(pd.DataFrame([4,5,4,4,5,5]), npartitions=2)
mi = privbayes.mutual_information(trues, preds)

mi = dask.compute(mi)[0]

print(mi)
'''