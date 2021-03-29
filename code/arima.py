#-------------------------------------------------------------------#
# Run arima
#-------------------------------------------------------------------#

import os
import numpy as np
import pandas as pd
import pmdarima as pm
from helpers import *
from case_def import *
from algo_output import *

def Arima(case, case_type):
    ''''Function for Arima'''
    size = len(case['data'])
    if size > 100:
        data_old = case['data']
        data = case['data'][size-100:]
    else:
        data_old = case['data']

    if size < 100:
        start_point = 0
    else:
        start_point = size - 100
    counter = np.arange(start_point, size, 1)

    df, df_train, df_test = train_test_split(data, counter)

    algo_output = {}

    ''' Arima model is called here. Parameters can be changed as needed'''
    step_model = pm.auto_arima(df_train['data'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                               start_P=0, seasonal=True, d=1, D=1, trace=False, approx=False,
                               error_action='ignore', supress_warnings=True,
                               c=False, disp=-1, stepwise=True)

    print('First arima model fitted ...')
    step_model.fit(df_train['data'])

    fit_forecast_pred = step_model.predict_in_sample(df_train['data'])
    fit_forecast = pd.DataFrame(fit_forecast_pred, index=df_train.index, columns=['Prediction'])

    fut_forecast_pred = step_model.predict(n_periods=len(df_test['data']))
    fut_forecast = pd.DataFrame(fut_forecast_pred, index=df_test.index, columns=['Prediction'])

    test_list = df_test['data'].values
    mse_test = (fut_forecast_pred - test_list)

    algo_case_dump = CreateAlgoResult(case_type)
    algo_case_dump.past_anoms(fut_forecast_pred, df_test)
    algo_case_dump.debug_creation(fut_forecast_pred, df_test)
    algo_case_dump.se_metric(df_test['data'].values, fut_forecast_pred)



    '''
    updated_model = step_model.fit(df['data'])
    pickle.dump(step_model, open('file_name', 'wb'))
    res_fol_path = os.path.join('results', 'saved_models')

    if case.meta.file_name.split('.')[0] in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), res_fol_path)):
        os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              res_fol_path,
                              case.meta.file_name.split('.')[0]))
        os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), res_fol_path, case.meta.file_name.split('.')[0]))
        file_name = case_type + '_' + case.meta.file_name.split('.')[0]
        pickle.dump(step_model, open(file_name, 'wb'))
    else:
        os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  res_fol_path))
        os.mkdir(case.meta.file_name.split('.')[0])
        os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              res_fol_path,
                              case.meta.file_name.split('.')[0]))
        file_name = case_type + '_' + case.meta.file_name.split('.')[0]
        pickle.dump(step_model, open(file_name, 'wb'))
        
    forecast = updated_model.predict(n_periods=fut_steps).tolist()

    algo_case_dump.forecast_creation(forecast, size, fut_steps)
    '''


    return algo_case_dump.algo_output





