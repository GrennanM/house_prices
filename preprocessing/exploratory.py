import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from graph_code import *
from scipy import stats
import seaborn as sns

# # read original training data
# dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
# dfTrain = pd.read_csv(dataTrain, index_col=0)
# print (dfTrain['OverallCond'].head())

# read original test data
dataTest = '/home/markg/kaggle/house_prices/data/original/test.csv'
dfTest = pd.read_csv(dataTest, index_col=0)
# print (dfTest.info())

# drawHistogram('LotFrontage')

# # read working train data
# dataTrain = '/home/markg/kaggle/house_prices/data/working/train_17_01_2019_1733.csv'
# dfTrain = pd.read_csv(dataTrain, index_col=0)
# print ('Train: ')
# print (dfTrain.info())
# print ('-'*30)

# # read working test dataset
# dt = '/home/markg/kaggle/house_prices/data/working/test_17_01_2019_1721.csv'
# test_df = pd.read_csv(dt, encoding='latin-1', index_col=0)
# print ('Test: ', test_df.info())

# count the number of different types of variables
# print (dfTrain.dtypes.value_counts())

# returns count of a categorical variable
# print (dfTrain['GarageCars'].value_counts())
# print (dfTrain['Neighborhood'].value_counts().index) # list of index in order

# # sort dataframe by another column
# print (dfTrain.groupby('MSZoning').count())

# # select features of a certain type
# for feature in dfTrain.select_dtypes(include=[object]):
#     drawBarchart(str(feature), title=str(feature) + "_barchart")

# # # print skewnewss and kurtosis
# print ("Skew: ", dfTrain['LotFrontage'].skew())
# print ("Kurt: ", dfTrain['LotFrontage'].kurt())

# # print scatterPlots of livingAreas
# prints scatterplots on top of one another
# livingAreas = ['GrLivArea', '1stFlrSF', '2ndFlrSF']
# for i in range(len(livingAreas)):
#     drawScatter(livingAreas[i], title=str(livingAreas[i]))

# # calculate rsquared
# dfTrain['squaredYearBuilt'] = dfTrain['YearBuilt']**2
# dfTrain['SalePrice'] = dfTrain['SalePrice'].apply(np.log)
# x = dfTrain['YearBuilt']
# y = dfTrain['SalePrice']
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# print ("r-squared: ", r_value**2)

# select an individual value from scatter plot YearBuilt
df1 = dfTrain[['LotFrontage', 'SalePrice']] # create new df with selected columns
print (df1.sort_values(by = 'LotFrontage'))
# dfTrain.drop([186], axis=0, inplace=True) # see YearBuilt scatterplot

# ################ Missing Values ################################
#
# # list of features with missing values
# featuresWithMissingValues = [feature for feature in dfTrain
#                                 if dfTrain[feature].isna().sum() != 0]
#
# dictFeaturesWithMissingValues = {feature: dfTrain[feature].isna().sum()
#                                     for feature in featuresWithMissingValues}
#
# # sorted column of features with the sum of their missing values
# totalNumberMissingValues = dfTrain.isna().sum().sort_values(ascending=False)
#
# # percentage of missing data
# percentMissingData = (dfTrain.isna().sum()/len(dfTrain.index)).sort_values(ascending=False)
#
# # table of missing data and percentage
# missingData = pd.concat([totalNumberMissingValues, percentMissingData],
#                         axis = 1, keys=['Total', 'Percentage'])
# print (missingData.head(50))

# ################# END Missing Values ##############################
