from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy
import time
import util
import classifier
import datetime

def main():

    stock_name = '