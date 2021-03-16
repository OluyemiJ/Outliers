#––-----------------------------------------------------------#
# Define the case
#-------------------------------------------------------------#
import os
import pandas as pd
from types import SimpleNamespace

class CaseDefinition:
    '''Case definition class to get details such as input file path, case type etc'''

    def __init__(self, case_type, cwd, sub_path, algo_dict):
        '''Class initialisation'''
        self.case_file = pd.DataFrame()
        self.case_file.meta = SimpleNamespace()
        self.algo_dict = algo_dict
        self.case_type = case_type
        self.working_dir = cwd
        self.folder_sub_path = sub_path
        self.extract_input_data()
        self.get_case_type()
        self.get_case_dir_path()
        self.get_res_dir_path()

    def __str__(self):
        '''Print objects of the class'''
        return str(self.__class__) + ':' + str(self.__dict__)

    def extract_input_data(self):
        '''Extract input data into case dataframe'''

        for file in os.listdir(self.get_case_dir_path()):
            file_path = os.path.join(self.get_case_dir_path(), file)
            if not file.startswith('.'):
                if file.endswith('.csv'):
                    raw_data = pd.read_csv(file_path)
                    self.case_file.meta.file_name = file
                elif file.endswith('.xlsx'):
                    raw_data = pd.read_excel(file_path)
                    self.case_file.meta.file_name = file

                for col in raw_data.columns:
                    if col == 'timestamp':
                        self.case_file['timestamp'] = raw_data[col]
                    elif col == 'data':
                        self.case_file['data'] = raw_data[col]

    def get_case_type(self):
        '''Returns the case type value(s)'''
        self.case_file.meta.algorithms_selected = []
        for i in self.case_type:
            self.case_file.meta.algorithms_selected\
                .append(self.algo_dict.get(i, 'Invalid algorithm'))

    def get_case_dir_path(self):
        '''Returns file case input data file path'''
        self.case_file.meta.case_file_dir = os.path.join(os.path.dirname(self.working_dir), 'cases', self.folder_sub_path)
        return self.case_file.meta.case_file_dir


    def get_res_dir_path(self):
        '''Returns file case input data file path'''
        self.case_file.meta.res_file_dir = os.path.join(os.path.dirname(self.working_dir), 'results', self.folder_sub_path)
