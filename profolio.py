from sklearn import preprocessing
import numpy
import datetime
import matplotlib.pyplot as plt

import util
import classifier

ETFs = ['XLE', 'XLU', 'XLK', 'XLB', 'XLP', 'XLY', 'XLI', 'XLV', 'SPY']

class Profolio:

    def __init__(self, name):
        self.name = name
        self.profits = numpy.zeros(252)

    def accProfits(self):
        return numpy.cumsum(self.profits)

    def annualSharpeRatio(self, n = 252):
        return numpy.sqrt(n) * self.profits.mean() / self.profits.std()

def smart_trade(etf, method, delta):
    parameters = [8, 0.0125]

    data = util.get_data(etf, '2014/1/1', '2016/12/31')

    # keep a copy for unscaled data for later gain calculation
    # TODO replace by MinMax_Scaler.inverse_transform()
    #
    # the first day of test is 2015/12/31. Using this data on this day to predict
    # Up/Down of 2016/01/04
    test = data[data.index > datetime.datetime(2015,12,30)]

    le = preprocessing.LabelEncoder()
    test['UpDown'] = (test['Close'] - test['Open']) / test['Open']
    threshold = 0.000
    test.UpDown[test.UpDown >= threshold] = 'Up'
    test.UpDown[test.UpDown < threshold] = 'Down'
    test.UpDown = le.fit(test.UpDown).transform(test.UpDown)
    test.UpDown = test.UpDown.shift(-1) # shift 1, so the y is actually next day's up/down

    dataMod = util.applyFeatures(data, range(1, delta))
    dataMod = util.preprocessData(dataMod)

    tr = dataMod[dataMod.index <= datetime.datetime(2015,12,30)]
    #tr = dataMod[dataMod.index <= datetime.datetime(2016,06,30)]
    te = dataMod[dataMod.index > datetime.datetime(2015,12,30)]
    te = te[te.columns[0:-1]] # remove Up/Down label from testing
    cl