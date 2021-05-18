
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
    features = dataset.columns[0:-1]

    if method == 'RNN':
        clf = performRNNlass(dataset[features], dataset['UpDown'])
        return clf

    elif method == 'RF':
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

    elif method == 'KNN':
        clf = neighbors.KNeighborsClassifier()

    elif method == 'SVM':
        c = parameters[0]
        g =  parameters[1]
        clf = SVC(C=c, gamma=g)

    elif method == 'ADA':
        clf = AdaBoostClassifier()

    return clf.fit(dataset[features], dataset['UpDown'])

def prepareDataForClassification(dataset, start_test):
    """
    generates categorical output column, attach to dataframe
    label the categories and split into train and test
    """
    features = dataset.columns[0:-1]

    X = dataset[features]
    y = dataset.UpDown

    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]

    X_test = X[X.index >= start_test]
    y_test = y[y.index >= start_test]

    return X_train, y_train, X_test, y_test


def performClassification(X_train, y_train, X_test, y_test, method, parameters, savemodel):
    """
    performs classification on returns using serveral algorithms
    """
    if method == 'RF':
        return performRFClass(X_train, y_train, X_test, y_test, parameters, savemodel)

    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test, parameters, savemodel)

    elif method == 'SVM':
        return performSVMClass(X_train, y_train, X_test, y_test, parameters, savemodel)

    elif method == 'ADA':
        return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters, savemodel)

    elif method == 'RNN':
        X_test = numpy.reshape(numpy.array(X_test), (X_test.shape[0], 1, X_test.shape[1]))

        model = performRNNlass(X_train, y_train)
        return model.evaluate(X_test, y_test)[1]
