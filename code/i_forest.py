#-------------------------------------------------------------------------------#
# Run isolation forest algorithm
#-------------------------------------------------------------------------------#
from helpers import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest as IF

class IsolationForest:

    def __init__(self, inp_data, truth):
        '''Initialise the class'''
        self.data = inp_data.extract_input_data()
        self.truth_val = truth
        self.get_score()

    def __str__(self):
        '''Produce human readable form of the class'''
        return str(self.__class__) + ':' + str(self.__dict__)

    def get_isolation_forest_outliers(self):
        '''Get outliers based on isolation forest'''
        for i, j in self.data.items():
            j['datetime'] = [pd.to_datetime(x) for x in j['timestamp']]
            db_df = pd.DataFrame(list(zip(j['datetime'], j['data'])), columns=['datetime', 'data'])
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

            self.data[i].update({'prediction': anom_bool.tolist()})

        return self.data

    def get_score(self):
        '''Get F1 Score and return'''
        score = f1_metric(self.get_isolation_forest_outliers(), self.truth_val, 'I_Forest')
        return score