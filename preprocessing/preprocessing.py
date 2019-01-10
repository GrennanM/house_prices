import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)

# convert response, LotArea and GrLivArea to log scale
dfTrain['SalePrice'] = dfTrain['SalePrice'].apply(np.log)
dfTrain['LotArea'] = dfTrain['LotArea'].apply(np.log)
dfTrain['GrLivArea'] = dfTrain['GrLivArea'].apply(np.log)

# # remove 3 outliers
# print (dfTrain.sort_values(by = 'GrLivArea', ascending=False)[:2]) # selects first two rows
dfTrain.drop([524, 1299], axis=0, inplace=True) # see GrLivArea scatterplot
dfTrain.drop([186], axis=0, inplace=True) # see YearBuilt scatterplot

######################## Missing Values ####################

# impute mode for single missing value in Electrical column
dfTrain['Electrical'].fillna(dfTrain['Electrical'].mode().iloc[0],
                            inplace=True)

# list of features with missing values
featuresWithMissingValues = [feature for feature in dfTrain
                                if dfTrain[feature].isna().sum() != 0]

# drop columns with missing values
dfTrain.drop(columns=featuresWithMissingValues, inplace=True)

######################## END Missing Values ####################

# YearBuilt
# create a column for square of year built (see YearBuilt scatter) & drop original
dfTrain['squaredYearBuilt'] = dfTrain['YearBuilt']**2
dfTrain.drop(columns=['YearBuilt'], inplace=True)

# bin squaredYearBuilt into 10 equal sized bins
dfTrain['squaredYearBuilt'] = pd.cut(dfTrain['squaredYearBuilt'], 10)

# # count number of entries in each bin
# print (dfTrain['squaredYearBuilt'].value_counts())

# get dummy variables for each bin
dfTrain = pd.get_dummies(dfTrain, columns=['squaredYearBuilt'], drop_first=True)

## Need to update below to include binned year. Start here!
# maybe include TotRmsAbvGrd
starterVars = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
                'FullBath', 'LotArea', 'YearBuilt', 'Neighborhood']

# create dataframe with just chosen features
dfTrain = dfTrain[starterVars]

# need to include YearBuilt. Currently here!!
catgVars = ['OverallQual', 'GarageCars', 'FullBath', 'Neighborhood']

# get dummy variables for catgeorical features
dfTrain = pd.get_dummies(dfTrain, columns=catgVars, drop_first=True)
print (dfTrain.info())
