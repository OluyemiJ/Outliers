#-----------------------------------------------------------------#
# Perform z-value detection on input data
#-----------------------------------------------------------------#
import numpy as np
from helpers import *

class ZValue:
    '''Z Value class'''

    def __init__(self, inp_data, truth):
        '''class initialisation'''
        self.data = inp_data.extract_input_data()
        self.truth_val = truth
        self.get_score()


    def __str__(self):
        '''Print objects of the class'''
        return str(self.__class__) + ':' + str(self.__dict__)

    def get_z_outliers(self):
        '''Get outliers based on z value scores'''
        for i, j in self.data.items():

            data_std = np.std(j['data'])
            data_mean = np.mean(j['data'])

            low_limit = data_mean - data_std*3
            high_limit = data_mean + data_std*3

            anom_bool = np.logical_or(j['data'] > high_limit, j['data'] < low_limit)
            anom_bool = np.where(anom_bool == True, 1, anom_bool)
            anom_bool = np.where(anom_bool == False, 0, anom_bool)

            self.data[i].update({'prediction': anom_bool.tolist()})

        return self.data

    def get_score(self):
        '''Get F1 Score and return'''
        score = f1_metric(self.get_z_outliers(), self.truth_val, 'Z_Value')
        return score


