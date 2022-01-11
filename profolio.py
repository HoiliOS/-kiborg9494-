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
        return numpy.sqrt(n) * self.profits.mean() / self.profit