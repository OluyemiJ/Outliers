# -------------------------------------------------------------#
# Extract the input data
# -------------------------------------------------------------#
import os
import pandas as pd


class InputData:
    '''Inputting the data class'''

    def __init__(self, case_desc):
        '''Class initialisation'''
        self.dir_path = case_desc.get_case_dir_path()
        self.extract_input_data()

    def extract_input_data(self):
        '''Extract input data into a dictionary'''
        proc_data = {}

        for file in os.listdir(self.dir_path):
            file_path = os.path.join(self.dir_path, file)
            if not file.startswith('.'):
                if file.endswith('.csv'):
                    raw_data = pd.read_csv(file_path)
                elif file.endswith('.xlsx'):
                    raw_data = pd.read_excel(file_path)

                raw_data_n = {}
                bool_changes = {True: 1,
                                False: 0}
                for col in raw_data.columns:
                    if col == 'label':
                        changes = [bool_changes.get(x) for x in list(raw_data[col])]
                    else:
                        changes = list(raw_data[col])


                    raw_data_n.update({col: changes})

                proc_data.update({file: raw_data_n})
            else:
                pass

        return proc_data
