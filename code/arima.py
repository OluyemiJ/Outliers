#-------------------------------------------------------------------#
# Run arima
#-------------------------------------------------------------------#

import os
import numpy as np
import pandas as pd
import pmdarima as pm
from helpers import *
from case_def import *
import pickle

def Arima(inp_data, fut_steps, name):
    ''''Function for Arima'''
    for i, j in inp_data.items():
        size = len(j['data'])
        if size > 100:
            data_old = j['data']
            data = j['data'][size-100:]
        else:
            data_old = j['data']

        if size < 100:
            start_point = 0
        else:
            start_point = size - 100
        counter = np.arange(start_point, size, 1)

        df, df_train, df_test = train_test_split(data, counter)

        arima_res = {}
        step_model = pm.auto_arima(df_train['Data'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                                   start_P=0, seasonal=True, d=1, D=1, trace=False, approx=False,
                                   error_action='ignore', supress_warnings=True,
                                   c=False, disp=-1, stepwise=True)

        print('First arima model fitted ...')
        step_model.fit(df_train['Data'])

        fit_forecast_pred = step_model.predict_in_sample(df_train['Data'])
        fit_forecast = pd.DataFrame(fit_forecast_pred, index=df_train.index, columns=['Prediction'])

        fut_forecast_pred = step_model.predict(n_periods=len(df_test['Data']))
        fut_forecast = pd.DataFrame(fut_forecast_pred, index=df_test.index, columns=['Prediction'])

        test_list = df_test['Data'].values
        mse_test = (fut_forecast_pred - test_list)

        mse, rmse, mae, smap, r2 = se_metric(df_test['Data'], fut_forecast_pred)

        print('Finished Arima anomaly ... Starting forecasting ...')

        updated_model = step_model.fit(df['Data'])

        res_filename = os.path.join(CaseDefinition.get_res_dir_path(), 'saved_models/arima')
        pickle.dump(step_model, open(res_filename, 'wb'))

        forecast = updated_model.predict(n_periods=fut_steps).tolist()

        j[i].update({'forecast': forecast})

    return inp_data





