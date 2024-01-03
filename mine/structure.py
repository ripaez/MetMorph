import pandas as pd
import numpy as np
from src.preprocess import *

#test()

## Preprocessing stage
path_to_data = 'data/adult.csv'
data = load(path_to_data, verbose=False)
data_train, data_test = split(data, size=0.2, verbose=True)
clean_train,discarded_train = clean(data_train, symbol=["?"," ?"], verbose=True)
