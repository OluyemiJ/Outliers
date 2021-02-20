#--------------------------------------------------------------#
#  Run Model
#--------------------------------------------------------------#

from run_mod import *
import os

# select mode to run
# 0: use Z Score and clustering methods to find outliers in the bulk data
# 1: use forecasting algorithms to find past and predict future anomalies
mode = 0

# select algorithm of interest for mode 0:
# -1: use all clustering algorithms
# 0: use z-test and all clustering algorithms
# 1: use only z-test algorithm
# 2: use dbscan algorithm
# 3: use isolation forest algorithm
case_type_prox = 0

# select algorithm of interest for mode 1:
# 0: use arima and other forecasting algorithms
# 1: use arima
# 2: use holtwinters
# 3: use LSTM
# 4: use vector autoregression
# 5: use temporal convolution
case_type_for = 0

# say whether your dataset has labels
# 0 for no labels (dataset is taken as truth) and 1 for labels. Comparing two or algos in mode 0 require labels
truth = 1

# define current working directory
cwd = os.getcwd()

# define case sub path
sub_path = 'lab/nab'

# execute model
run_model(mode, case_type_prox, truth, cwd, sub_path)

