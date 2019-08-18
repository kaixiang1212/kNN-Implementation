'''
Data Preprocessor:
	This file contains all function requies for data preprocessing
	- Load data sheet
	- Split into target
	- Split into train and test set
	- Trim missing data
	- Calculate percentage error
	- Calculate error for discrete prediction
'''

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Load file depending on the input
def load_datasheet(filename, separator = ',', hd = None, use_col = []):
    if use_col:
        df =  pd.read_csv(filename,sep=separator,header = hd,usecols=use_col)
    else:
        df = pd.read_csv(filename,sep = separator, header = hd)
    return df

# split data into feature set and target set
def split_target(df):
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return x, y

# split data into train and test set by ratio
def split_train_test(df,ratio):
    size = int(len(df)*ratio)
    train = df.iloc[:size,:]
    test = df.iloc[size:,:]
    return train, test

# data normalisation
def normalise_data(df):
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

# trim data contain '?' and convert string number to numeric format
def trim_data(df):
    df = df[(df.astype(str) != '?').all(axis=1)]
    df = df.apply(pd.to_numeric)
    return df
