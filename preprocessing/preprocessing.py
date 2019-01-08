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

# # remove 2 outliers (see GrLivArea scatterplot)
# print (dfTrain.sort_values(by = 'GrLivArea', ascending=False)[:2])
dfTrain.drop([524, 1299], axis=0, inplace=True)

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
