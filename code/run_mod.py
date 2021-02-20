# --------------------------------------------------------------------#
# Run Model
# --------------------------------------------------------------------#

import os
from case_def import *
from input_data import *
from z_val import *
from scores import *
from db_scan import *
from i_forest import *
from arima import *

print('Starting ...')


def run_model(mode, case_type_prox, truth, cwd, sub_path):
    '''Run the model'''

    case_desc = get_case_desc(mode, case_type_prox, cwd, sub_path)
    inp_data = get_input_data(case_desc)

    result = get_run_algos(case_desc, truth, inp_data)

    # print(result)
    # make result
    print('Done ... OK')
    print(f'Please collect results from {os.path.dirname(cwd)}/results/{sub_path}')

    return result


def get_case_desc(mode, case_type_prox, cwd, sub_path):
    '''Define the case'''
    print('Defining the case ....')
    case_desc = CaseDefinition(mode, case_type_prox, cwd, sub_path)
    return case_desc


def get_input_data(case_desc):
    '''Get the input data'''
    print('Getting the data ...')
    inp_data = InputData(case_desc)
    return inp_data


def get_run_algos(case_desc, truth, inp_data):
    '''Get and run algorithms'''
    case_type_list = case_desc.get_case_type()[0]
    result = {}
    if isinstance(case_type_list, list):
        for i in case_type_list:
            print(f'Now running {i} ...')
            res = switch_algorithms(i, truth, inp_data)
            result.update({i: res})
    else:
        print(f'Now running {case_type_list} ...')
        res = switch_algorithms(case_type_list, truth, inp_data)
        result.update({case_type_list: res})

    return result


def switch_algorithms(algo_text, truth, inp_data):
    '''Define switcher for algorithms to get right function'''
    switcher = {
        'Z Value': run_z_value(truth, inp_data),
        'DB Scan': run_db_scan(truth, inp_data),
        'Isolation Forest': run_isolation_forest(truth, inp_data)
    }
    return switcher.get(algo_text, 'Invalid algorithm')


# runners for the algortihms
# z value
def run_z_value(truth, inp_data):
    '''Run the z-value algorithm'''
    all_res = {}
    data_res = ZValue(inp_data, truth).get_z_outliers()
    z_val_res = ZValue(inp_data, truth).get_score()
    all_res.update({'Data': data_res, 'Scores': z_val_res})
    return all_res


# density_scan
def run_db_scan(truth, inp_data):
    '''Run the db_scan algorithm'''
    all_res = {}
    data_res = DbScan(inp_data, truth).get_db_scan_outliers()
    db_scan_res = DbScan(inp_data, truth).get_score()
    all_res.update({'Data': data_res, 'Scores': db_scan_res})
    return all_res


# isolation_forest
def run_isolation_forest(truth, inp_data):
    '''Run isolation forest algorithm'''
    all_res = {}
    data_res = IsolationForest(inp_data, truth).get_isolation_forest_outliers()
    i_forest_res = IsolationForest(inp_data, truth).get_score()
    all_res.update({'Data': data_res, 'Scores': i_forest_res})
    return all_res

# arima
def run_arima_model(inp_data, truth):
    '''Run arima mode detect and forecast'''
    all_res = {}
    data_res = Arima(inp_data, truth)
    print(data_res)

# holt_winter()
# lstm()
# vector_autoregression()
