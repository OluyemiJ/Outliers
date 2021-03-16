#-----------------------------------------------------------------#
# Perform z-value detection on input data
#-----------------------------------------------------------------#
import numpy as np

def get_z_outliers(case, case_type):
    '''Get outliers based on z value scores'''

    data_std = np.std(case['data'])
    data_mean = np.mean(case['data'])

    low_limit = data_mean - data_std*3
    high_limit = data_mean + data_std*3

    anom_bool = np.logical_or(case['data'] > high_limit, case['data'] < low_limit)
    anom_bool = np.where(anom_bool == True, 1, anom_bool)
    anom_bool = np.where(anom_bool == False, 0, anom_bool)

    case[f'anomaly prediction {case_type}'] = anom_bool.tolist()

    #print(case.columns)
    return case



