from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy
import time
import util
import classifier
import datetime

def main():

    stock_name = 'SPY'
    delta = 4
    start = datetime.datetime(2010,1,1)
    end = datetime.datetime(2015,12,31)
    start_test = datetime.datetime(2015,1,1)

    dataset = util.get_data(stock_name, start, end)
    delta = range(1, delta)
    dataset = util.applyFeatures(dataset, delta)
    dataset = util.preprocessData(dataset)
    X_train, y_train, X_test, y_test  = \
        classifier.prepareDataForClassification(dataset, start_test)

    X_train = numpy.reshape(numpy.array(X_tr