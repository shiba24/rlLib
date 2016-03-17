import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class FcNN3(chainer.Chain):
    """ Fully-connected neural network of 3 layers. """
    def __init__(self, n_in, n_units, n_out):
        super(FcNN3, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
       )

    def __call__(self, x, t, index):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        h = F.select_item(h, index)

        self.loss = F.mean_squared_error(h, t)
        return self.loss

    def predict(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        return h




"""

#

#      # h_act = np.array([h.data[x, index[x]] for x in range(len(index))]).astype(np.float32)

        # self.accuracy = F.accuracy(h, t)

        # print(h_act)
        # print(h.data)
        # self.loss = F.mean_squared_error(chainer.Variable(h_act), t)

"""
