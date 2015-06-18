"""

Abstract class of  regression & classification model

Author: Shawn Liu
Date:   06-03-2015

"""

import abc

class Classifier(object):
    __metaclass__ = abc.ABCMeta
    isRegr        = None

    @abc.abstractmethod
    def build_classifier(self, instances):
        pass


    @abc.abstractmethod
    def classify_instances(self, instances):
        pass


    @abc.abstractmethod
    def classify_instance(self, instance):
        pass


    def isRegression(self):
        return self.isRegr