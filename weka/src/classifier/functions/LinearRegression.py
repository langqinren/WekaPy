"""

Multi-variate Linear Regression (OLS Ordinary Least Square) Implementation

Author:     Shawn Liu
Date:       06-03-2015

Reference:  http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html
Reading:    http://www.r-tutor.com/elementary-statistics/simple-linear-regression
            http://blog.minitab.com/blog/adventures-in-statistics/regression-analysis-tutorial-and-examples


##############
# Definition #
##############
The classic linear regression (i.e. the estimation method OLS) has some assumptions:
(1)
(2)
The OLS can perform poorly if multicollinearity (X_i and X_j are highly correlated) is present in data, unless the sample
size is large.


##############
# Estimation #
##############
A number of methods have been developed for parameter estimation and inference in linear regression
(1) Least Square Estimation
    (a) ordinary least square (OLS)
    (b) generalized least square (GLS)

(2) Maximum Likelihood Estimation
    (a) MLE
    (b) Ridge regression
    (c) least absolute deviation

(3) Bayesian Linear Regression


#############
# Extension #
#############
Several extension of linear regression have been made:

(1) general linear model - Y is multi-variate
(2) generalized linear model
(3) heteroscedastic model
(4) hierarchical linear model
(5) errors-in-variables


############
# Question #
############

Q1: Does regression need feature selection?
A1: http://www.math.upatras.gr/~dany/Downloads/hercma07.pdf
    http://www.simafore.com/blog/bid/111764/How-to-apply-feature-selection-in-linear-regression-with-RapidMiner

Q2: How to compare two regression models?
A2:

"""


from ..Classifier import Classifier

import numpy as np

class LinearRegression(Classifier):
    __model = None

    def __init__(self):
        self.isRegr = True

    """
        Y = X * P
    """
    def build_classifier(self, instances):
        X = instances.getFeature()
        X = self._appendOnes(X)
        Y = instances.getClass()
        #self._OLS(X, Y)
        self._batchGradientDescent(X, Y)

    def classify_instances(self, instances):
        X = instances.getFeature()
        X = self._appendOnes(X)
        if self.__model is None:
            raise Exception('I know Python!')
        else:
            return np.dot(X, self.__model)


    def classify_instance(self, instance):
        X = instance.getFeature()
        X = self._appendOnes(X)
        if self.__model is None:
            raise Exception('I know Python!')
        else:
            return np.dot(X, self.__model)

    def _appendOnes(self, X):
        B = np.ones((X.shape[0],1))
        X = np.append(X, B, 1)
        return X


    def _OLS(self, X, Y):
        """
         Model 1: without bias Y = X * P
        """
        # do not modify X

        """
         Model 2: with bias Y = X * P + E
        """
        self.__model = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
        print self.__model

    """
        Gradient descent method is a way to find a local minimum of a function. The algorithm will converge where the
        gradient is zero.
        http://www.onmyphd.com/?p=gradient.descent
    """
    def _batchGradientDescent(self, X, Y):
        n = X.shape[0]
        m = X.shape[1]

        B  = np.zeros((m,1))
        _B = np.zeros((m,1))
        alpha = 0.0001

        for t in range(1):

            for j in range(0, m,1):
                G = self._computeGradient(X, Y, B, j)
                _B[j] = B[j] - alpha * G

        self.__model = B
        print self.__model


    def _computeGradient(self, X, Y, B, j):
        n = X.shape[0]
        m = X.shape[1]

        delta = 0
        for i in range(0,m,1):
            delta += (np.dot(X[i:i+1,:], B) - Y[i]) * X[i, j]
        return delta / n


    """
        http://research.microsoft.com/pubs/192769/tricks-2012.pdf
    """
    def _stochasticGradientDescent(self):
        pass