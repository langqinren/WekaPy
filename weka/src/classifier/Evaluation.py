"""

How to compare regression model: http://people.duke.edu/~rnau/compare.htm

"""

import numpy as np
import math

class Evaluation(object):
    _predict    = None
    _truth      = None
    _isRegr     = None
    _size       = None

    def evaluateModel(self, classifier, test):
        self._isRegr  = classifier.isRegression()
        self._truth   = test.getClass()
        self._predict = classifier.classify_instances(test)
        self._size    = test.numOfInstance()

    def crossValidateModel(self, classifier, data, nfold):
        pass

    def printSummaryString(self):

        if self._isRegr:
            print "Correlation Coefficient:     %f" % self.correlationCoefficient()
            print "Mean Absolute Error:         %f" % self.meanAbsoluteError()
            print "Root Mean Squared Error:     %f" % self.rootMeanSquaredError()
            print "Relative Absolute Error:     %f" % self.relativeAbsoluteError()
            print "Root Relative Squared Error: %f" % self.rootRelativeSquaredError()
            print "Total Number of Instances:   %d" % self._size
        else:
            print "cls"

    def correlationCoefficient(self):
        cof = np.corrcoef(self._truth, self._predict, rowvar=0)
        return cof[0][1]

    def meanAbsoluteError(self):
        return np.sum(np.abs(self._truth - self._predict)) / self._size


    def rootMeanSquaredError(self):
        return np.linalg.norm(self._truth - self._predict) / math.sqrt(self._size)


    def relativeAbsoluteError(self):
        m = np.mean(self._truth)
        e1 = np.sum(np.abs(self._truth - self._predict))
        e2 = np.sum(np.abs((self._truth - m)))
        return e1 / e2


    def rootRelativeSquaredError(self):
        m = np.mean(self._truth)
        e1 = np.linalg.norm(self._truth - self._predict)
        e2 = np.linalg.norm(self._truth - m)
        return e1 / e2


    def rSquared(self):
        pass