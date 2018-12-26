import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)

# store response
y = dfTrain['SalePrice']

def drawBarchart(column, title):
    # create a list of categories and values
    categories = [i for i in dfTrain[column].value_counts().index]
    values = [v for v in dfTrain[column].value_counts()]

    plt.bar(categories, values, align='center', alpha=0.5)
    plt.title(title)
    plt.savefig("/home/markg/kaggle/house_prices/graphs/barcharts/" + title +
                "_barchart.png")

def drawHistogram(column, title):
    binwidth=None
    binwidth = [x for x in range(0,500000, 10000)] # optional: set binwidth

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
    plt.scatter(dfTrain[column], y)
    # plt.xlim(0, 50000) # optional: set x axis limit
    plt.title(title + "_Vs_SalePrice")
    plt.savefig("/home/markg/kaggle/house_prices/graphs/scatterPlots/" + title +
                "_scatter.png")

def drawBoxplot(column, title):
    plt.boxplot(dfTrain[column])
    plt.title(title)
    plt.savefig("/home/markg/kaggle/house_prices/graphs/boxplots/" + title +
                "_boxplot.png")
