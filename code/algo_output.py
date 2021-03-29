# Create the results for the algorithms
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def smape(a, f):
    return 100/len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))

class CreateAlgoResult:

    def __init__(self, algo_name):
        self.algo_name = algo_name
        self.algo_output = {}
        self.algo_output['Algorithm Name'] = algo_name

    def past_anoms(self, forecast, df_test):
        df_past_anoms = pd.DataFrame(df_test['count'], index=df_test.index, columns=['expected value'])
        df_past_anoms.rename(columns={df_test.columns[0]: "step"}, inplace=True)
        df_past_anoms['expected value'] = forecast

        df_past_anoms['real_value'] = df_test['data']
        test_list = df_test['data'].values

        mse = mean_squared_error(test_list, forecast)
        rmse = np.sqrt(mse)
        df_past_anoms['step'] = df_test['count']

        df_past_anoms['mse'] = mse
        df_past_anoms['rmse'] = rmse

        df_past_anoms['mae'] = mean_absolute_error(test_list, forecast)
        df_past_anoms['anomaly_score'] = abs(df_past_anoms['expected value'] - df_past_anoms['real_value']) / df_past_anoms['mae']

        df_past_anoms_ult = df_past_anoms[:30]

        df_past_anoms_ult = df_past_anoms_ult[(df_past_anoms_ult.index > ((df_past_anoms.index.max()) - 30))]

        if len(df_past_anoms_ult) == 0:
            exists_anom_last = 'FALSE'
        else:
            exists_anom_last = 'TRUE'



        df_past_anoms = df_past_anoms[(df_past_anoms['anomaly_score'] > 2)]

        max = df_past_anoms['anomaly_score'].max()
        min = df_past_anoms['anomaly_score'].min()
        df_past_anoms['anomaly_score'] = (df_past_anoms['anomaly_score'] - min) / (max - min)


        max = df_past_anoms_ult['anomaly_score'].max()
        min = df_past_anoms_ult['anomaly_score'].min()

        df_past_anoms_ult['anomaly_score'] = (df_past_anoms_ult['anomaly_score'] - min) / (max - min)

        self.algo_output['present_status'] = exists_anom_last
        self.algo_output['present_alerts'] = df_past_anoms_ult.fillna(0).to_dict(orient='record')
        self.algo_output['past'] = df_past_anoms.fillna(0).to_dict(orient='record')

        return ('ok')

    def forecast_creation(self, forecast, start_step, fut_steps):
        df_future = pd.DataFrame(forecast, columns=['value'])
        df_future.rename(columns={df_future.columns[0]: "value"}, inplace=True)

        df_future['value'] = df_future.value.astype("float32")

        df_future['step'] = np.arange(start_step, start_step + fut_steps, 1)
        self.algo_output['future'] = df_future.to_dict(orient='record')

        return ('OK')

    def debug_creation(self, trained_data, df_test):
        testing_data = pd.DataFrame(trained_data, index=df_test.index,
                                    columns=['value'])  # ,columns=['expected value'])
        testing_data.rename(columns={"yhat": "expected value"}, inplace=True)

        testing_data['step'] = testing_data.index
        self.algo_output['debug'] = testing_data.to_dict(orient='record')

    def se_metric(self, test_list, list_yhat):
        mse = mean_squared_error(test_list, list_yhat)
        rmse = np.sqrt(mse)
        self.algo_output['mae'] = mean_absolute_error(test_list, list_yhat)
        self.algo_output['rmse'] = rmse
        self.algo_output['mse'] = mse
        self.algo_output['smap'] = smape(test_list, list_yhat)
        self.algo_output['r2'] = r2_score(test_list, list_yhat)

