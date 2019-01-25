import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime

# read original Test data
dataTest = '/home/markg/kaggle/house_prices/data/original/test.csv'
dfTest = pd.read_csv(dataTest)

# convert response, LotArea and GrLivArea to log scale
dfTest['LotArea'] = dfTest['LotArea'].apply(np.log)
dfTest['GrLivArea'] = dfTest['GrLivArea'].apply(np.log)

# impute for missing values
dfTest['TotalBsmtSF'].fillna(dfTest['TotalBsmtSF'].mean(), inplace=True)
dfTest['GarageCars'].fillna(dfTest['GarageCars'].median(), inplace=True)
dfTest['Functional'].fillna(dfTest['Functional'].mode(), inplace=True)

# starter variables
starterVars = ['Id', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
                'FullBath', 'LotArea', 'YearBuilt', 'Neighborhood', 'TotRmsAbvGrd',
                'OverallCond']

# create dataframe with just chosen features
dfTest = dfTest[starterVars]

# topcode TotRmsAbvGrd, FullBath and GarageCars
dfTest.loc[dfTest['TotRmsAbvGrd'] > 12, 'TotRmsAbvGrd'] = 12
dfTest.loc[dfTest['FullBath'] > 3, 'FullBath'] = 3
dfTest.loc[dfTest['GarageCars'] > 3, 'GarageCars'] = 3

# convert GarageCars to integer so it's the same type as in training data
dfTest['GarageCars'] = dfTest['GarageCars'].astype(int)

# create a column for square of year built (see YearBuilt scatter) & drop original
dfTest['squaredYearBuilt'] = dfTest['YearBuilt']**2
dfTest.drop(columns=['YearBuilt'], inplace=True)

# bin squaredYearBuilt into 10 equal sized bins
binLabels = [i for i in range(10)]
dfTest['squaredYearBuilt'] = pd.cut(dfTest['squaredYearBuilt'],
                                    10, labels=binLabels)

# categorical variables
catgVars = ['OverallQual', 'GarageCars', 'FullBath', 'Neighborhood',
                'squaredYearBuilt', 'TotRmsAbvGrd', 'OverallCond']

# get dummy variables for catgeorical features
dfTest = pd.get_dummies(dfTest, columns=catgVars, drop_first=True)

# standardize numeric variables
numeric = ['GrLivArea', 'LotArea']
dfTest[numeric] = preprocessing.StandardScaler().fit_transform(dfTest[numeric])

# write working Test data to csv
TestingPath = '/home/markg/kaggle/house_prices/data/working/'
dataTestFilename = 'test_' + str(datetime.now().strftime('%d_%m_%Y_%H%M')) + '.csv'
dfTest.to_csv(TestingPath + dataTestFilename)

# # print data info
# print ("Test dataset: ")
# print (dfTest.info())
# print("-"*30)
