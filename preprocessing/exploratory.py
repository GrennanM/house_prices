import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)

# count the number of different types of variables
# print (dfTrain.dtypes.value_counts())

# # # list of features with missing values
# featuresWithMissingValues = [feature for feature in dfTrain
#                                 if dfTrain[feature].isna().sum() != 0]
#
# dictFeaturesWithMissingValues = {feature: dfTrain[feature].isna().sum()
#                                     for feature in featuresWithMissingValues}

# # prints features with missing values and the count of missing values
# for feature in dfTrain:
#     if dfTrain[feature].isna().sum() != 0:
#         print (str(feature) + ": " + str(dfTrain[feature].isna().sum()))

# print (dfTrain['GarageQual'].value_counts())
# print (dfTrain['GarageCond'].value_counts())

# print (dfTrain['OverallCond'].value_counts())
# print (dfTrain['OverallQual'].value_counts())
#
# plt.hist(dfTrain['OverallCond'], alpha=0.5, label='OverallCond')
# plt.hist(dfTrain['OverallQual'], alpha=0.5, label='OverallQual')
# plt.legend(loc='upper right')
# plt.savefig("/home/markg/kaggle/house_prices/graphs/OverallCond_OverallQual_histogram.png")
