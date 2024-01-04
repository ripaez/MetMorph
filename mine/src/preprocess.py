import os
import pandas as pd 
import numpy as np
import itertools
import collections
from sklearn.model_selection import train_test_split

def test():
    """
    Testing command
    """
    print('Hello world')
    return 

def load(rel_data_path: str, verbose : bool):
    """ Given the relative path from structure file to the dataset, the function loads the data to a Pandas dataframe
    """
    current_path = os.getcwd()
    try:
        df =pd.read_csv(os.path.join(current_path,rel_data_path))
        if len(df.columns)==1:
            raise TypeError("File found, delimiter badly define for csv. Correct delimited in function load.") 
        if verbose:
            print(df.head())
        print('Data loaded')
        return df
    except ValueError:
        print("The file is invalid")
        return None
    except FileNotFoundError:
        print(f"The file does not exist")
        return None

def split(dataframe: pd.DataFrame, size: float, verbose: bool = False):
    """ Tran and test sets splitting with ScikitLearn according to size definition. Random state 42 always.
    """
    try:
        train_set, test_set = train_test_split(dataframe, test_size=size, random_state=42)
        if verbose:
            print(f'Train set size: {len(train_set)} entries')
            print(f'Train set size: {len(test_set)} entries')
        return train_set, test_set
    except:
        print('Weird error going on')
        return None

def clean(input_df: pd.DataFrame, symbol=None, verbose:bool = False):
    """ The function eliminates entries with NaN. I also eliminates columns that only contain NaNs. 
    Additional non-allowed symbols are introduced as list in symbol var.
    Returns the clean dataframe and the list of eliminated columns (bool list). 
    """
    eliminated = input_df.isnull().all()
    input_df=input_df.loc[:,~eliminated] #Here it eliminates columns with full NANs
    if symbol:
        for i in symbol:
            for j in input_df.columns.to_list():
                input_df=input_df[input_df[j] != i] #Here it drops elements contained in the list of symbols
    input_df=input_df[input_df.index.notnull()] #Here it eliminates entries with missing indexing
    input_df = input_df.loc[~input_df.index.duplicated(keep='first')] #eliminate duplicated indexes its kept the first
    if verbose:
        print('Cleaning done')
    return input_df, eliminated

def labeling(input_df,label=None, eliminated=None):
    """ It refines from the column label list those that were removed during the cleaning,
    and updates the dataframe metadata
    """
    if len(eliminated) != 0:
        label=list(itertools.compress(label,~eliminated))
    if len(label) != 0 :
        input_df.columns = label
    return input_df

def encoding_prep(dataframe: pd.DataFrame, cat_columns: list, num_columns: list, neglected_columns: list,verbose=False):
    if len(neglected_columns) > 0:
        for ii in neglected_columns:
            if ii in cat_columns:
                cat_columns.remove(ii)
            elif ii in num_columns:
                num_columns.remove(ii)
            else:
                raise TypeError(f"In function 'encoding_prep' the neglected column '{ii}' defined do not belong to the dataframe or it is repeated as neglected.")
    total_labels = sorted([j for i in [cat_columns,num_columns,neglected_columns] for j in i])
    original_labels = sorted(dataframe.columns.tolist())
    if total_labels != original_labels:
        if len(total_labels) > len(original_labels):
            raise TypeError('Error: Possible double encoding categorization for columns in function EncodingPrep.')
        elif len(total_labels) < len(original_labels):
            raise TypeError('Error: Possible missing encoding categorization for columns in function EncodingPrep.')
        else:
            raise TypeError('Possible double/missing encoding caracterization happening. Check role of all columns.')
    else: 
        print('Encoding preparation done')   
        return cat_columns,num_columns,neglected_columns