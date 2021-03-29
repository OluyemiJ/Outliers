#-------------------------------------------------------------#
# Some helper functions
#-------------------------------------------------------------#
import numpy as np
import pandas as pd
import statsmodels.api as sm

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

def seasonal_options (a):
  print(" Starting seasonal finding")
  print(a)
  x =sm.tsa.stattools.pacf(a)

  possible =[]
  for i in range(4, len(x)-6):
    before2 = x[i-2]
    before= x[i-1]
    period = x[i]
    last = x[i+1]
    last2 = x[i+2]
    if (before2 < before < period > last ):
      possible.append(i-1)
  print ("Finishing seasonal finding")
  return possible






