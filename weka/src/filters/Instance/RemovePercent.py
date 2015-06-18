"""

"""

import numpy as np
import math

from ..Filter import Filter
from data.Instances import Instances


class RemovePercent(Filter):
    _percent = None

    def __init__(self, percent):
        self._percent = percent

    def setPercent(self, percent):
        if (percent > 0) and (percent < 1):
            self._percent = percent
        else:
            raise Exception("percentage is out of range (0, 1)")

    def process(self, instances):
        super(RemovePercent, self).process(instances)

        data   = instances.getData()
        clsIdx = instances.getClassIndex()

        indices = np.random.permutation(data.shape[0])
        split_idx =  int(math.floor(indices.shape[0] * self._percent))

        train_idx = indices[:split_idx]
        test_idx  = indices[split_idx:]

        train = Instances( data[train_idx,:], clsIdx )
        test  = Instances( data[test_idx,:],  clsIdx )

        if data is None:
            raise Exception("data hasn't been set, cannot apply filter")

        if self._percent is None:
            raise Exception("percent hasn't been set, cannot apply filter")

        return (train, test)