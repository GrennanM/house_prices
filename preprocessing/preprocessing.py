import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)

# # create n-1 dummy variables for 'Embarked' variable
# dfTrain = pd.get_dummies(dfTrain, columns=['MSSubClass'], prefix=['SubClass'],
#  drop_first=True)
# print (dfTrain.info())
