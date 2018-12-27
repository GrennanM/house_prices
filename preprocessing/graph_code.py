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
# dfTrain['SalePrice'] = dfTrain['SalePrice'].apply(np.log) # convert to log scale
dfTrain['LotArea'] = dfTrain['LotArea'].apply(np.log) # convert to log scale

# store response
y = dfTrain['SalePrice']

def drawBarchart(column, title):
    sns.catplot(x=column, kind="count", data=dfTrain)
                # order=list(dfTrain[column].value_counts().index),
                # height=8.27, aspect=11.7/8.27)
    # plt.xticks(rotation=50)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/markg/kaggle/house_prices/graphs/barcharts/" + title +
                "_barchart.png")

def drawViolinplot(column, title):
    # order Neighborhood by median SalePrice
    # result = dfTrain.groupby([column])['SalePrice'].aggregate(np.median).reset_index().sort_values('SalePrice',
    #                         ascending=False)
    sns.violinplot(x=column, y='SalePrice', data=dfTrain)
                # order=result[column])
    sns.set(style="whitegrid")
    # plt.xticks(rotation=50)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/markg/kaggle/house_prices/graphs/violinplots/" + title +
                "_Violinplot.png")

def drawHistogram(column, title):
    binwidth=None
    # binwidth = [x for x in range(10, 14, 0.2)] # optional: set binwidth

    plt.hist(dfTrain[column], alpha=0.5, bins=binwidth, label=column)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig("/home/markg/kaggle/house_prices/graphs/histograms/" + title +
                "_histogram.png")

def drawTwoHist(colA, colB, title):
    # plots a double histogram with overlaps
    plt.hist(dfTrain[colA], alpha=0.5, label=colA)
    plt.hist(dfTrain[colB], alpha=0.5, label=colB)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig("/home/markg/kaggle/house_prices/graphs/histograms/" + title +
                "_histogram.png")

def drawScatter(column, title):
    sns.scatterplot(x=column, y="SalePrice", data=dfTrain)

    # plt.xlim(0, 50000) # optional: set x axis limit
    plt.title(title + "_Vs_SalePrice", fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/markg/kaggle/house_prices/graphs/scatterPlots/" + title +
                "_scatter.png")

def drawBoxplot(column, title):
    plt.boxplot(dfTrain[column])
    plt.title(title)
    plt.savefig("/home/markg/kaggle/house_prices/graphs/boxplots/" + title +
                "_boxplot.png")
