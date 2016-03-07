import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class MLP(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_out),
#            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x, t, index):
        h = F.relu(self.l1(x))
#        h = F.relu(self.l2(h))
        h = self.l2(h)
        h_act = np.array([h.data[x, index[x]] for x in range(len(index))]).astype(np.float32)
        # print(h_act)

        self.loss = F.mean_squared_error(chainer.Variable(h_act), t)
        # print(self.loss.data)
        return self.loss


    def predict(self, x):
        h = F.relu(self.l1(x))
#        h = F.relu(self.l2(h))
        h = self.l2(h)

        return h

        # self.accuracy = F.accuracy(h, t)
