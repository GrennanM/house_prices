import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)

# convert response to log scale
dfTrain['SalePrice'] = dfTrain['SalePrice'].apply(np.log)

dfTrain['LotArea'] = dfTrain['LotArea'].apply(np.log) # convert to log scale

# create n-1 dummy variables for 'MSSubClass' variable
dfTrain = pd.get_dummies(dfTrain, columns=['MSSubClass'], prefix=['SubClass'],
 drop_first=True)
print (dfTrain.info())
