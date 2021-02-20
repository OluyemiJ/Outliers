#-------------------------------------------------------------#
# Some helper functions
#-------------------------------------------------------------#
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def se_metric(test_list, list_yhat):
    mse = mean_squared_error(test_list, list_yhat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_list, list_yhat)
    smap = smape(test_list, list_yhat)
    r2 = r2_score(test_list, list_yhat)

    return mse, rmse, mae, smap, r2


def f1_metric(data, truth_value, method):
    '''Calculate F1 scores for classification based algorithms'''
    res_score = {}

    for i, j in data.items():
        if truth_value == 0:
            res_score.update({i: {f'{method} F1 Score': 'No scores'}})
        else:
            f1 = f1_score(j['label'], j[f'{list(j.keys())[-1]}'], labels=np.unique(j[f'{list(j.keys())[-1]}']))
            res_score.update({i: {f'{method} F1 Score': f1}})
    return res_score


def train_test_split(data, counter):
    '''Create train test split'''
    df = pd.DataFrame()
    df['Count'] = counter
    df['Data'] = data

    df.set_index('Count', inplace=True, drop=False)
    test_start = len(df)*0.8

    train_set = df[:test_start]
    test_set = df[test_start:]
    return df, train_set, test_set

def smape(a, f):
    return 100/len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))




