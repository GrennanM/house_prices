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
# print (dfTrain['MiscFeature'].value_counts())
# print (dfTrain['Neighborhood'].value_counts().index) # list of index in order

# # sort dataframe by another column
# print (dfTrain.groupby('MSZoning').count())

# # select features of a certain type
# for feature in dfTrain.select_dtypes(include=[object]):
#     drawBarchart(str(feature), title=str(feature) + "_barchart")

# # print skewnewss and kurtosis
# print ("Skew: ", dfTrain['TotalBsmtSF'].skew())
# print ("Kurt: ", dfTrain['TotalBsmtSF'].kurt())

# # print scatterPlots of livingAreas
# prints scatterplots on top of one another
# livingAreas = ['GrLivArea', '1stFlrSF', '2ndFlrSF']
# for i in range(len(livingAreas)):
#     drawScatter(livingAreas[i], title=str(livingAreas[i]))

# # calculate rsquared
# dfTrain['GrLivArea'] = dfTrain['GrLivArea'].apply(np.log)
# dfTrain['SalePrice'] = dfTrain['SalePrice'].apply(np.log)
# x = dfTrain['GrLivArea']
# y = dfTrain['SalePrice']
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# print ("r-squared: ", r_value**2)

# ################ working on below....
# print ("Original: ", dfTrain['YearBuilt'].head(10))
dfTrain['squaredYearBuilt'] = dfTrain['YearBuilt']**2
# print ("-"*20)
# print ("Squared: ", dfTrain['squaredYearBuilt'].head(10))

## working on binning squared year built 
print (pd.cut(dfTrain['squaredYearBuilt'], 10))

# ################# Missing Values ################################
#
# # list of features with missing values
# featuresWithMissingValues = [feature for feature in dfTrain
#                                 if dfTrain[feature].isna().sum() != 0]
#
# dictFeaturesWithMissingValues = {feature: dfTrain[feature].isna().sum()
#                                     for feature in featuresWithMissingValues}
#
# sorted column of features with the sum of their missing values
# totalNumberMissingValues = dfTrain.isna().sum().sort_values(ascending=False)
#
# # percentage of missing data
# percentMissingData = (dfTrain.isna().sum()/len(dfTrain.index)).sort_values(ascending=False)
#
# # table of missing data and percentage
# missingData = pd.concat([totalNumberMissingValues, percentMissingData],
#                         axis = 1, keys=['Total', 'Percentage'])
# print (missingData.head(20))

# ################# END Missing Values ##############################
