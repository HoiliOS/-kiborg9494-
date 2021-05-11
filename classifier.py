
import numpy

from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def buildModel(dataset, method, parameters):
    """
    Build final model for predicting real testing data
    """