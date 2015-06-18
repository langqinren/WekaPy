"""

"""
import abc
import numpy as np

class Filter(object):
    __metaclass__ = abc.ABCMeta
    data          = None

    @abc.abstractmethod
    def process(self, instances):
        np.random.shuffle(instances.getData())

    @staticmethod
    def useFilter(instances, filter):
        return filter.process(instances)

