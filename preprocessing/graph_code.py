import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime

# read original train data
dataTrain = '/home/markg/kaggle/house_prices/data/original/train.csv'
dfTrain = pd.read_csv(dataTrain, index_col=0)

def drawBarchart(column, title):
    # create a list of categories and values
    categories = [i for i in dfTrain[column].value_counts().index]
    values = [v for v in dfTrain[column].value_counts()]

    plt.bar(categories, values, align='center', alpha=0.5)
    plt.title(title)
    plt.savefig("/home/markg/kaggle/house_prices/graphs/barcharts/" + title + ".png")

def drawTwoHist(colA, colB, title):
    # plots a double histogram with overlaps
    plt.hist(dfTrain[colA], alpha=0.5, label=colA)
    plt.hist(dfTrain[colB], alpha=0.5, label=colB)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig("/home/markg/kaggle/house_prices/graphs/histograms/" + title + ".png")

def drawHistogram(column, title):
    plt.hist(dfTrain[column], alpha=0.5, label=column)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig("/home/markg/kaggle/house_prices/graphs/histograms/" + title + ".png")
