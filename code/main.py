#--------------------------------------------------------------#
#  Run Model
#--------------------------------------------------------------#

from run_mod import *

# select algorithm of interest. Can use more than one algorithm:
# 1: use only z-test algorithm
# 2: use dbscan algorithm
# 3: use isolation forest algorithm
# 4: use arima
case_type = [4]

# define case sub path. i.e where is your input data relative from the cases folder
sub_path = 'lab/nab'

# execute model
run_model(case_type, sub_path)

