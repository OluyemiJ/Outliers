# --------------------------------------------------------------------#
# Run Model
# --------------------------------------------------------------------#

from case_def import *
from z_val import *
from scores import *
from db_scan import *
from i_forest import *
from arima import *

import os

print('Starting ...')

cwd = os.getcwd()

algo_dict = {1: 'Z Value', 2: 'DB Scan', 3: 'Isolation Forest', 4: 'ARIMA', 5: 'Holtwinters', 6: 'LSTM',
             7: 'Vector Auto Regression', 8: 'Temporal Convolution'}

def run_model(case_type, sub_path):
    '''Run the model'''

    case_dict = CaseDefinition(case_type, cwd, sub_path, algo_dict).case_file

    all_results = get_run_algos(case_dict)

    #print(all_results['ARIMA'].keys())
    print(all_results)
    # make result
    print('Done ... OK')
    print(f'Please collect results from {case_dict.meta.res_file_dir}')



def get_run_algos(case_dict):
    '''Get and run algorithms'''
    result = {}
    if len(case_dict.meta.algorithms_selected) > 1:
        for i in case_dict.meta.algorithms_selected:
            print(f'Now running {i} ...')
            res = switch_algorithms(case_dict, i)
            #print(res[f'anomaly prediction {i}'].value_counts())
            result.update({i: res})
            #print(result[i].columns)
    else:
        algo = case_dict.meta.algorithms_selected[0]
        print(f'Now running {algo} ...')
        res = switch_algorithms(case_dict, algo)
        #print(res[f'anomaly prediction {algo}'].value_counts())
        result.update({algo: res})

    return result


def switch_algorithms(case, case_type):
    '''Define switcher for algorithms to get right function'''
    if case_type == 'Z Value':
        run = get_z_outliers(case, case_type)
    elif case_type == 'DB Scan':
        run = get_db_scan_outliers(case, case_type)
    elif case_type == 'Isolation Forest':
        run = get_isolation_forest_outliers(case, case_type)
    elif case_type == 'ARIMA':
        run = Arima(case, case_type, fut_steps=5)

    return run