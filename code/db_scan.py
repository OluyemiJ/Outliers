# -------------------------------------------------------------------------------#
# Run the db-scan
# -------------------------------------------------------------------------------#
from helpers import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np


class DbScan:
    '''Db Scan class'''

    def __init__(self, inp_data, truth):
        self.data = inp_data.extract_input_data()
        self.truth_val = truth
        self.get_score()

    def __str__(self):
        '''Print objects of the class'''
        return str(self.__class__) + ':' + str(self.__dict__)

    def get_db_scan_outliers(self):
        '''Get outliers based on db_scan'''
        for i, j in self.data.items():
            j['datetime'] = [pd.to_datetime(x) for x in j['timestamp']]
            db_df = pd.DataFrame(list(zip(j['datetime'], j['data'])), columns=['datetime', 'data'])
            db_df['hour'] = db_df['datetime'].dt.hour
            db_df_n = db_df[['hour', 'data']]
            db_df_n_vals = db_df_n.values.astype('float32', copy=False)

            db_df_scaler = StandardScaler().fit(db_df_n_vals)
            db_df_n_vals = db_df_scaler.transform(db_df_n_vals)

            model_db = DBSCAN(eps=0.3, metric='euclidean').fit(db_df_n_vals)
            db_df['labels'] = model_db.labels_
            db_df['labels'].mask(db_df['labels'] != -1, 0, inplace=True)
            db_df['labels'].replace(-1, 1, inplace=True)


            anom_bool = db_df['labels'].values

            self.data[i].update({'prediction': anom_bool.tolist()})

        return self.data

    def get_score(self):
        '''Get F1 Score and return'''
        score = f1_metric(self.get_db_scan_outliers(), self.truth_val, 'DB_Scan')
        return score
