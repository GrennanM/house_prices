import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from graph_code import *
from scipy import stats
import seaborn as sns

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)
# print (dfTrain.info())

# # count the number of different types of variables
# print (dfTrain.dtypes.value_counts())

# # returns count of a categorical variable
print (dfTrain['FullBath'].value_counts())
# print (dfTrain['Neighborhood'].value_counts().index) # list of index in order

# # sort dataframe by another column
# print (dfTrain.groupby('MSZoning').count())

# # select features of a certain type
# for feature in dfTrain.select_dtypes(include=[object]):
#     drawBarchart(str(feature), title=str(feature) + "_barchart")

# # print skewnewss and kurtosis
# print ("Skew: ", dfTrain['LotArea'].skew())
# print ("Kurt: ", dfTrain['LotArea'].kurt

# # print scatterPlots of livingAreas
# # prints scatterplots on top of one another
# livingAreas = ['GrLivArea', '1stFlrSF', '2ndFlrSF','TotalBsmtSF']
# for i in range(len(livingAreas)):
#     drawScatter(livingAreas[i], title=str(livingAreas[i]))

# drawViolinplot('FullBath', title='FullBath')
# drawBarchart('FullBath', title='FullBath')

# # calculate rsquared
# x = dfTrain['GrLivArea']
# y = dfTrain['SalePrice']
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# print ("r-squared: ", r_value**2)

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
# drawHistogram('SalePrice', title='SalePrice')
