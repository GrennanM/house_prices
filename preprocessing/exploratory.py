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

# count the number of different types of variables
# print (dfTrain.dtypes.value_counts())

# returns count of a categorical variables
# print (dfTrain['MSZoning'].value_counts())

# sort dataframe by another column
# print (dfTrain.groupby('MSZoning').count())

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
# drawBarchart('MSZoning', title = "MSZoning_Barchart2")
# drawHistogram('LotArea', title="LotArea_histogram")

# # create n-1 dummy variables for 'Embarked' variable
# dfTrain = pd.get_dummies(dfTrain, columns=['MSSubClass'], prefix=['SubClass'],
#  drop_first=True)
# print (dfTrain.info())
