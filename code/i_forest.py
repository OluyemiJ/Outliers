#-------------------------------------------------------------------------------#
# Run isolation forest algorithm
#-------------------------------------------------------------------------------#
from helpers import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest as IF


def get_isolation_forest_outliers(case, case_type):
    '''Get outliers based on isolation forest'''
    db_df = case[['timestamp', 'data']].copy()
    db_df['datetime'] = pd.to_datetime(db_df['timestamp'])
    db_df['hour'] = db_df['datetime'].dt.hour
    db_df_n = db_df[['hour', 'data']]
    db_df_n_vals = db_df_n.values.astype('float32', copy=False)

    db_df_scaler = StandardScaler().fit(db_df_n_vals)
    db_df_n_vals = db_df_scaler.transform(db_df_n_vals)

    model_if = IF(contamination=0.05).fit(db_df_n_vals)
    db_df['labels'] = model_if.fit_predict(db_df_n_vals)
    db_df['labels'].mask(db_df['labels'] != -1, 0, inplace=True)
    db_df['labels'].replace(-1, 1, inplace=True)

    anom_bool = db_df['labels'].values
    case[f'anomaly prediction {case_type}'] = anom_bool.tolist()

    return case

