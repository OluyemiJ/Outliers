#-------------------------------------------------------------#
# Some helper functions
#-------------------------------------------------------------#
import numpy as np
import pandas as pd


def train_test_split(data, counter):
    '''Create train test split'''
    df = pd.DataFrame()
    df['count'] = counter
    df['data'] = data.values

    df.set_index('count', inplace=True, drop=False)
    test_start = len(df)*0.8

    print(f'train length is {test_start}')
    train_set = df[:round(test_start)]
    test_set = df[round(test_start):]

    return df, train_set, test_set






