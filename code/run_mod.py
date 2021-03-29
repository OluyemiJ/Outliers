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
import pandas as pd

print('Starting ...')

cwd = os.getcwd()

algo_dict = {1: 'Z Value', 2: 'DB Scan', 3: 'Isolation Forest', 4: 'ARIMA'}

def run_model(case_type, sub_path):
    '''Run the model'''

    case_dict = CaseDefinition(case_type, cwd, sub_path, algo_dict).case_file

    all_results = get_run_algos(case_dict)
    all_results_df = pd.DataFrame(all_results)
    print(all_results)
    # make result
    print('Done ... OK')
    # print(f'Please collect results from {case_dict.meta.res_file_dir}')



def get_run_algos(case_dict):
    '''Get and run algorithms'''
    result = {}
    if len(case_dict.meta.algorithms_selected) > 1:
        for i in case_dict.meta.algorithms_selected:
            print(f'Now running {i} ...')
            res = switch_algorithms(case_dict, i)
            result.update({i: res})
    else:
        algo = case_dict.meta.algorithms_selected[0]
        print(f'Now running {algo} ...')
        res = switch_algorithms(case_dict, algo)
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
        run = Arima(case, case_type)

    return run