import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import preprocessing
from datetime import datetime
import seaborn as sns

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)

# convert response, LotArea and GrLivArea to log scale
dfTrain['SalePrice'] = dfTrain['SalePrice'].apply(np.log)
dfTrain['LotArea'] = dfTrain['LotArea'].apply(np.log)
dfTrain['GrLivArea'] = dfTrain['GrLivArea'].apply(np.log)

# removing outliers
dfTrain.drop([524, 1299], axis=0, inplace=True) # see GrLivArea scatterplot
dfTrain.drop([692, 1183], axis=0, inplace=True)

# store response
y = dfTrain['SalePrice']

def drawBarchart(column):
    sns.catplot(x=column, kind="count", data=dfTrain)
                # order=list(dfTrain[column].value_counts().index),
                # height=8.27, aspect=11.7/8.27)
    plt.xticks(rotation=50)
    plt.title(column, fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/markg/kaggle/house_prices/graphs/barcharts/" + column +
                "_barchart.png")

def drawViolinplot(column):
    # order Neighborhood by median SalePrice
    # result = dfTrain.groupby([column])['SalePrice'].aggregate(np.median).reset_index().sort_values('SalePrice',
    #                         ascending=False)
    sns.violinplot(x=column, y='SalePrice', data=dfTrain)
                # order=result[column])
    sns.set(style="whitegrid")
    # plt.xticks(rotation=50)
    plt.title(column, fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/markg/kaggle/house_prices/graphs/violinplots/" + column +
                "_Violinplot.png")

def drawHistogram(column):
    binwidth=None
    # binwidth = [x for x in range(10, 14, 0.2)] # optional: set binwidth

    plt.hist(dfTrain[column], alpha=0.5, bins=binwidth, label=column)
    plt.title(column)
    plt.legend(loc='upper right')
    plt.savefig("/home/markg/kaggle/house_prices/graphs/histograms/" + column +
                "_histogram.png")

def drawDistplot(column):
    # adds a distribution plot on top of histogram
    sns.distplot(dfTrain[column], label=column)
    plt.title(column)
    plt.legend(loc='upper right')
    plt.savefig("/home/markg/kaggle/house_prices/graphs/histograms/" + column +
                "_histogram.png")

def drawTwoHist(colA, colB, title):
    # plots a double histogram with overlaps
    plt.hist(dfTrain[colA], alpha=0.5, label=colA)
    plt.hist(dfTrain[colB], alpha=0.5, label=colB)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig("/home/markg/kaggle/house_prices/graphs/histograms/" + title +
                "_histogram.png")

def drawScatter(column):
    # # to include a regression line
    # sns.lmplot(x=column, y="SalePrice", data=dfTrain)

    # to include just scatterplot with legend
    sns.scatterplot(x=column, y="SalePrice", data=dfTrain, label=column)
    # plt.xlim(0, 50000) # optional: set x axis limit
    # plt.legend(loc='upper right')

    plt.title(column + "_Vs_SalePrice", fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/markg/kaggle/house_prices/graphs/scatterPlots/individual/"
                + column + "_scatter.png")

def drawHeatmap():
    corrmat = dfTrain.corr()

    # Generate a mask to remove upper triangle
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, mask=mask, square=True, cmap="YlGnBu",
                linewidths=.5)
    plt.tight_layout()
    plt.savefig("/home/markg/kaggle/house_prices/graphs/scatterPlots/heatmap.png")

def drawBoxplot(column):
    plt.boxplot(dfTrain[column])
    plt.title(column)
    plt.savefig("/home/markg/kaggle/house_prices/graphs/boxplots/" + column +
                "_boxplot.png")

# # zoomed in heatmap
# k = 10 # number of variables for heatmap
# corrmat = dfTrain.corr()
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(dfTrain[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
#     annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.tight_layout()
# plt.savefig("/home/markg/kaggle/house_prices/graphs/scatterPlots/heatmap_zoomed.png")

# # # draw pairplot
# sns.pairplot(data=dfTrain, vars=['SalePrice', 'OverallQual', 'GrLivArea',
#                         'GarageCars', 'FullBath', 'LotArea', 'YearBuilt'],
#                         height=1.8)
# plt.tight_layout()
# plt.savefig("/home/markg/kaggle/house_prices/graphs/scatterPlots/pairplot.png")
