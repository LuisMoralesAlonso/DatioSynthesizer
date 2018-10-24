import dask
import dask.dataframe as df
import pandas as pd
import json
import datiosynthesizer.config as config
import datiosynthesizer.utils as utils
import datiosynthesizer.attributes as attributes

def init_random(seed: int = 0) -> None:
    utils.set_random_seed(seed)

def init_generator(file_desc):
    init_random(config.seed)
    return None