import pandas as pd
import numpy as np
from src.preprocess import *
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

#test()

## Preprocessing stage
path_to_data = 'data/adult.csv'
data = load(path_to_data, verbose=False)
data_train, data_test = split(data, size=0.2, verbose=True)

clean_train,train_discarded_cols = clean(data_train, symbol=["?"," ?"], verbose=True)
clean_train = labeling(clean_train,label=data.columns,eliminated=train_discarded_cols)

non_used_columns = ['fnlwgt']
non_num_columns = clean_train.select_dtypes(exclude=['number','bool_']).columns.to_list()
num_columns = clean_train.select_dtypes(['number','bool_']).columns.to_list()

non_num_columns,num_columns,non_used_columns = encoding_prep(clean_train,non_num_columns,num_columns,non_used_columns,verbose=False)

print(non_num_columns)
print(num_columns)

clean_train.drop(columns=non_used_columns,inplace=True)
preprocessing = make_column_transformer((StandardScaler(),num_columns),(OneHotEncoder(),non_num_columns))
encoded_train = preprocessing.fit_transform(clean_train)

print(encoded_train.shape)
print(preprocessing.get_feature_names_out())