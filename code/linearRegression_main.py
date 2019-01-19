import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from datetime import datetime

# train dataset
dataset = '/home/markg/kaggle/house_prices/data/working/train_17_01_2019_1733.csv'
df = pd.read_csv(dataset, index_col=0)

# split into 80% training, 20% testing
y = df['SalePrice']
X = df.drop(columns = ['SalePrice'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

# build linear regression model
# regr = linear_model.LinearRegression()

# train model
# regr.fit(X_train, y_train)

# # # cross-validation
# scores = cross_val_score(regr, X, y, cv=10, n_jobs=-1)
# print ("Accuracy: ", scores.mean())

############################ submission ################################
# test dataset
dt = '/home/markg/kaggle/house_prices/data/working/test_19_01_2019_0805.csv'
test_df = pd.read_csv(dt, encoding='latin-1', index_col=0)

# prepare Id for submission file
Id = test_df['Id']
test_df.drop(columns = ['Id'], inplace = True)

# fit linear regression for submission
regr = linear_model.LinearRegression()
regr.fit(X, y)
predictions = regr.predict(test_df)
predictions = np.exp(predictions) # convert from logs back to original scale

# create submission file
submission = pd.DataFrame({'Id':Id, 'SalePrice':predictions})
path = '/home/markg/kaggle/house_prices/data/submissions/'
filename = 'linearRegression_submission_' + str(datetime.now().strftime('%d_%m_%Y_%H%M')) + '.csv'
submission.to_csv(path+filename, index=False)
