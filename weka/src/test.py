from classifier.functions.LinearRegression import LinearRegression

from filters.Filter import Filter
from filters.Instance.RemovePercent import RemovePercent
from data.Instances import Instances
from classifier.Evaluation import Evaluation

import numpy as np

def readData():
    file = "../data/ex3x.csv";
    data = np.genfromtxt(file, skip_header=1, delimiter=',')
    return data


data    = Instances(readData())
filter  = RemovePercent(0.6)
(train, test) = Filter.useFilter(data, filter)

model = LinearRegression()
model.build_classifier(train)

eval = Evaluation()
eval.evaluateModel(model, test)
eval.printSummaryString()