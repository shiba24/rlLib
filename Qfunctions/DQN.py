import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from BaseClass import QfunctionBase

import NeuralNet
import chainer
from chainer import optimizers


class DQN(QfunctionBase):
    def __init__(self, discretize=10, gpu=-1):
        self.discretize = discretize
        self.gpu = gpu
        if self.gpu >= 0:
            from chainer import cuda
            self.xp = cuda.cupy
            cuda.get_device(self.gpu).use()
        else:
            self.xp = np

    def __call__(self, inputs):
        if len(self.xp.array(inputs).shape) == 1:
            n = 1
        else:
            n = len(inputs)
        inputs_var = chainer.Variable(self.xp.asarray(inputs).astype(np.float32).reshape(n, 2))
        output = self.rawFunction.predict(inputs_var).data
        if self.gpu < 0:
            return output.transpose()
        else:
            return self.xp.asnumpy(output.transpose())

    """ You need to call this function for the first time. """
    def initialize(self, Agent, n_hidden=50):
        self.n_input = len(Agent.state)
        self.n_out = len(Agent.actionlist)
        self.n_hidden = n_hidden
        self.rawFunction = NeuralNet.FcNN3(self.n_input, self.n_hidden,
                                           self.n_out)
        if self.gpu >= 0:
            self.rawFunction.to_gpu()

    def setupOptimizerSGD(self, lr=0.00025, momentum=0.95):
        self.optimizer = optimizers.MomentumSGD(lr=lr, momentum=momentum)
        self.optimizer.setup(self.rawFunction)


    def update(self, X, y, batchsize):
        i = 2
        Xstate = X[:, 0:i] # / 10.0 - 0.5
        y_scaled = y # / 10.0

        # training, random sampling of the size of minibatch-size
        sum_loss = 0
        perm = np.random.permutation(len(X))
        ind = 0
        x = chainer.Variable(self.xp.asarray(Xstate[perm[ind:ind + batchsize]]))
        t = chainer.Variable(self.xp.asarray(y_scaled[perm[ind:ind + batchsize]]))
        indexes = chainer.Variable(self.xp.asarray(X[perm[ind:ind + batchsize], i].astype(np.int32)))
        self.optimizer.update(self.rawFunction, x, t, indexes)
        sum_loss += float(self.rawFunction.loss.data) * len(t.data)

    def copy(self, originalDQN):
        self.n_input = originalDQN.n_input
        self.n_out = originalDQN.n_out
        self.n_hidden = originalDQN.n_hidden
        self.rawFunction = originalDQN.rawFunction.copy()
        if self.gpu >= 0:
            self.rawFunction.to_gpu()
