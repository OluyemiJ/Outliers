#––-----------------------------------------------------------#
# Define the case
#-------------------------------------------------------------#

import os

class CaseDefinition:
    '''Case definition class to get details such as input file path, case type etc'''

    def __init__(self, mode, case_type_prox, cwd, sub_path):
        '''Class initialisation'''
        self.mode_bool = mode
        self.case_type_prox_no = case_type_prox
        self.working_dir = cwd
        self.folder_sub_path = sub_path
        self.get_case_type()
        self.get_case_dir_path()

    def __str__(self):
        '''Print objects of the class'''
        return str(self.__class__) + ':' + str(self.__dict__)

    def get_case_type(self):
        '''Returns the case type value(s)'''

        if self.mode_bool == 0:
            case_dict_clust = {1: 'Z Value',
                               2: 'DB Scan',
                               3: 'Isolation Forest',
                               0: ['Z Value', 'DB Scan', 'Isolation Forest']}

            case = [case_dict_clust[self.case_type_prox_no]]

        elif self.mode_bool == 1:
            case_dict_fore = {1: 'Arima',
                              2: 'LSTM',
                              3: 'Vector Auto Regression',
                              4: 'Holtwinters',
                              0: ['Arima', 'LSTM', 'Vector Auto Regression', 'Holtwinters']}

            case = [case_dict_fore[self.case_type_prox_no]]

        return case

    def get_case_dir_path(self):
        '''Returns file case input data file path'''
        case_file_dir = os.path.join(os.path.dirname(self.working_dir), 'cases', self.folder_sub_path)

        return case_file_dir

    def get_res_dir_path(self):
        '''Returns file case input data file path'''
        res_file_dir = os.path.join(os.path.dirname(self.working_dir), 'results', self.folder_sub_path)

        return res_file_dir
