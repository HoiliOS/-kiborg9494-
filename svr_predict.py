
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
