import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from graph_code import *

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)
# print (dfTrain.info())
# print (dfTrain['MSSubClass'])

# count the number of different types of variables
# print (dfTrain.dtypes.value_counts())

# returns count of a categorical variables
# print (dfTrain['MSZoning'].value_counts())

# sort dataframe by another column
# print (dfTrain.groupby('MSZoning').count())

# # select features of a certain type
# for feature in dfTrain.select_dtypes(include=[object]):
#     drawBarchart(str(feature), title=str(feature) + "_barchart")

# ################# Missing Values ################################
#
# # list of features with missing values
# featuresWithMissingValues = [feature for feature in dfTrain
#                                 if dfTrain[feature].isna().sum() != 0]
#
# dictFeaturesWithMissingValues = {feature: dfTrain[feature].isna().sum()
#                                     for feature in featuresWithMissingValues}
#
# # prints features with missing values and the count of missing values
# for feature in dfTrain:
#     if dfTrain[feature].isna().sum() != 0:
#         print (str(feature) + ": " + str(dfTrain[feature].isna().sum()))
#
# ################# END Missing Values ##############################

# drawTwoHist('OverallCond', 'OverallQual', title="OverallCond_OvverallQual_histogram")
drawHistogram('SalePrice', title='SalePrice')
