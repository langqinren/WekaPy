"""

"""
import numpy as np

class Instances:
    _data    = None
    _insNum  = None
    _atrrNum = None
    _clsIdx  = None

    def __init__(self, data, clsIdx = None):
        self._data    = data
        self._insNum  = data.shape[0]
        self._atrrNum = data.shape[1]-1

        if clsIdx is None:
            clsIdx = data.shape[1]-1

        self._clsIdx = clsIdx


    def addData(self, data):
        self._data = data


    def addInstance(self, instance):
        if self._data is None:
            self._data = np.empty((0, instance.shape[1]))
        self._feature = np.vstack((self._data, instance))


    def getData(self):
        return self._data


    def getInstance(self, index):
        return self._data[index]


    def getClassIndex(self):
        return self._clsIdx


    def getFeature(self):
        return np.delete(self._data, self._clsIdx, 1)


    def getClass(self):
        return self._data[:,self._clsIdx : self._clsIdx+1]

    def numOfInstance(self):
        return self._insNum

    def numOfAttribute(self):
        return self._atrrNum