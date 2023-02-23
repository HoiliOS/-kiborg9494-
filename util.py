
# Ref: http://francescopochetti.com/stock-market-prediction-part-ii-feature-generation/
from sklearn import preprocessing
import pandas_datareader.data as web
import pandas as pd
import classifier
import numpy as np
import operator

pd.options.mode.chained_assignment = None

def preprocessData(dataset):

    le = preprocessing.LabelEncoder()

    # in case divid-by-zero
    dataset.Open[dataset.Open == 0] = 1

    # add prediction target: next day Up/Down
    threshold = 0.000
    dataset['UpDown'] = (dataset['Close'] - dataset['Open']) / dataset['Open']
    dataset.UpDown[dataset.UpDown >= threshold] = 'Up'
    dataset.UpDown[dataset.UpDown < threshold] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    dataset.UpDown = dataset.UpDown.shift(-1) # shift 1, so the y is actually next day's up/down
    dataset = dataset.drop(dataset.index[-1]) # drop last one because it has no up/down value
    return dataset

def count_missing(dataframe):
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()

def performCV(X_train, y_train, folds, method, parameters, savemodel):
    """
    given complete dataframe, number of folds, the % split to generate
    train and test set and features to perform prediction --> splits
    dataframein test and train set. Takes train set and splits in k folds.
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    returns mean of test accuracies
    """
    print ''
    print 'Parameters --------------------------------> ', parameters