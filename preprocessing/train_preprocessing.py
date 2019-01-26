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

# # remove outliers
# print (dfTrain.sort_values(by = 'GrLivArea', ascending=False)[:2]) # selects first two rows
dfTrain.drop([524, 1299], axis=0, inplace=True) # see GrLivArea scatterplot
dfTrain.drop([692, 1183], axis=0, inplace=True) # remove GrLivArea values larger than 4000, from Author's recomendations
dfTrain.drop([186], axis=0, inplace=True) # see YearBuilt scatterplot

# impute mode for single missing value in Electrical column
dfTrain['Electrical'].fillna(dfTrain['Electrical'].mode().iloc[0],
                            inplace=True)

# starter variables
starterVars = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
                'FullBath', 'LotArea', 'YearBuilt', 'Neighborhood', 'SalePrice',
                'TotRmsAbvGrd', 'OverallCond', 'BsmtFinType1']

# create dataframe with just chosen features
dfTrain = dfTrain[starterVars]

# topcode TotRmsAbvGrd and GarageCars
dfTrain.loc[dfTrain['TotRmsAbvGrd'] > 12, 'TotRmsAbvGrd'] = 12
dfTrain.loc[dfTrain['TotRmsAbvGrd'] == 3, 'TotRmsAbvGrd'] = 4
dfTrain.loc[dfTrain['GarageCars'] > 3, 'GarageCars'] = 3

# create a column for square of year built (see YearBuilt scatter) & drop original
dfTrain['squaredYearBuilt'] = dfTrain['YearBuilt']**2
dfTrain.drop(columns=['YearBuilt'], inplace=True)

# bin squaredYearBuilt into 10 equal sized bins
binLabels = [i for i in range(10)]
dfTrain['squaredYearBuilt'] = pd.cut(dfTrain['squaredYearBuilt'],
                                    10, labels=binLabels)

# categorical variables
catgVars = ['OverallQual', 'GarageCars', 'FullBath', 'Neighborhood',
                'squaredYearBuilt', 'TotRmsAbvGrd', 'OverallCond']

# get dummy variables for catgeorical features
dfTrain = pd.get_dummies(dfTrain, columns=catgVars, drop_first=True)

# get dummy for BsmtFinType1 including a column for NAs
dfTrain = pd.get_dummies(dfTrain, columns=['BsmtFinType1'], drop_first=True,
                        dummy_na=True)

# standardize numeric variables
numeric = ['GrLivArea', 'LotArea', 'TotalBsmtSF']
dfTrain[numeric] = preprocessing.StandardScaler().fit_transform(dfTrain[numeric])

# write working training data to csv
trainingPath = '/home/markg/kaggle/house_prices/data/working/'
dataTrainFilename = 'train_' + str(datetime.now().strftime('%d_%m_%Y_%H%M')) + '.csv'
dfTrain.to_csv(trainingPath + dataTrainFilename, index=True)

# # print data info
# print ("Train dataset: ")
# print (dfTrain.info())
# print("-"*30)
