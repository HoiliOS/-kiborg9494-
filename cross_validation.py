import util
import datetime

def main():

    stock_name = 'SPY'
    method = 'KNN'

    maxdeltas = 99 # min is 3
    folds = 10

    start = datetime.datetime(2014,1,1)
    end = datetime.datetime(2015,12,31)
    start_test = datetime.datetime(2015,1,1)

    # UNCOMMENT to do Feature selection
    # parameters = [16, 0.125]
   