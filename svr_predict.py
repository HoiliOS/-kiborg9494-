
# REF: https://github.com/llSourcell/predicting_stock_prices/demo.py
import sys
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


class SVR_Solver:

    def __init__(self):

        self.dates = []
        self.prices = []

    def get_data(self, filename):

        with open(filename, 'r') as csvfile:
            csvFileReader = csv.reader(csvfile)
            next(csvFileReader)     # skipping column names
            for row in csvFileReader:
                self.prices.append(float(row[1]))

        dates = range(1, len(self.prices)+1)
        self.dates = np.reshape(dates, (len(dates), 1))

        cutpoint = len(self.prices)/10

        self.train_date = self.dates[cutpoint+1:]
        self.train_price = self.prices[cutpoint+1:]

        self.test_date = self.dates[:cutpoint]
        self.test_price = self.prices[:cutpoint]

    def training(self, c, g):

        self.model = SVR(kernel= 'rbf', C= c, gamma= g)
        self.model.fit(self.train_date, self.train_price) # fitting the data points in the models

    def tune_parameter(self, c_range, g_range):

        best = (None, None, 9999999)
        for c in xrange(c_range[0], c_range[1]+1):
            for g in xrange(g_range[0], g_range[1]+1):
                self.training(2**c, 2**g)
                pred = self.predict(self.test_date)
                mse = self.MSE(pred, self.prices)
                if mse < best[2]:
                    best = (c, g, mse)
                print c, g, mse