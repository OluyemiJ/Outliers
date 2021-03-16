# -------------------------------------------------------------------------------#
# Run the db-scan
# -------------------------------------------------------------------------------#
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np


def get_db_scan_outliers(case, case_type):
    '''Get outliers based on db_scan'''
    db_df = case[['timestamp', 'data']].copy()
    db_df['datetime'] = pd.to_datetime(db_df['timestamp'])
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

    case[f'anomaly prediction {case_type}'] = anom_bool.tolist()

    return case
