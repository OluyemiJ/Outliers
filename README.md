# Outliers
## Unitesting
## Front end
## Backend

## Detector
This app holds the contents of a simple anomaly detector. It's aim is to understand the behaviour of a univariate time series dataset.
It features 7 algorithms which use a combination of simple parametric, proximity based methods and more complex prediction methods. There are 4 methods is total:
1. Z Test (Standard deviation method)
2. DB Scan (Density based clustering)
3. Isolation Forest (Datapoints partitioning)
4. LSTM (Long short term memory)
5. LSTM (Long short term memory) Autoencoders
6. Holt Winters
7. ARIMA

These algorithms all have different strengths and weaknesses depending on the model of the data being considered.
The app compares these algorithms using two metrics the MAE and the .. MAE was used for the supervised algorithms while 
... was used for the others. The best algorithm is selected for the supervised and unsupervised algorithms.

a. The point is identified as an anomaly by at least 2 out of 4 algorithms  
b. The point is identified an an anomaly by the LSTM Autoencoder

More theoretical infomation on the algorithms listed above can be found here:
- https://towardsdatascience.com/z-score-for-anomaly-detection-d98b0006f510
- https://towardsdatascience.com/density-based-algorithm-for-outlier-detection-8f278d2f7983

## Data sources
https://github.com/numenta/NAB/tree/master/data

# Future work
**Improve scoring system for unsupervised algos**

**Implement python api for easy integration in jupyter notebooks and other applications**

**Backend database**

**Human feedback system**

**online/streaming anomaly detection**

# Getting started guide
