import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import NeuralNet
import chainer
xp = np

class Qfunction(object):
    """
    Qfunction class for generalized tasks of q-learning.
    This class should be argument of Agent.takeAction(qFunction, epsilon, gamma)
        qFunction can be both np.ndarray, or function object.
    """

    def __init__(self):
        pass


class Qfield(Qfunction):
    def __init__(self):
        pass

"""
    def initializeQ(self, size):
        self.q_field = np.random.random(np.append(size, len(self.actionlist))) * 0.1 + 0.5

    # Q_new = reward + gamma max_a' Q(s', a')
    def updateQ(self, reward, gamma):
        target = reward + gamma * np.max(self.q_field[tuple(self.memory_state[-1])])
        diff = target - self.q_field[tuple(np.append(self.memory_state[-2], self.memory_act[-1]))]
        self.q_field[tuple(np.append(self.memory_state[-2], self.memory_act[-1]))] += self.stepsizeparameter * diff
"""



class DQN(Qfunction):
    def __init__(self, gpu=-1):
        self.gpu = gpu
        if self.gpu >= 0:
            from chainer import cuda
            xp = cuda.cupy
            cuda.get_device(args.gpu).use()


    def __call__(self, inputs):
        if len(xp.array(inputs).shape) == 1:
            n = 1
        else:
            n = len(inputs)
        inputs_var = chainer.Variable(xp.asarray(inputs).astype(np.float32).reshape(n, 2))
        output = self.rawFunction.predict(inputs_var).data
        if self.gpu < 0:
            return output.transpose()
        else:
            return xp.asnumpy(output.transpose())


    """ You need to call this function for the first time. """
    def initialize(self, Agent, n_hidden=50):
        self.n_input = len(Agent.state)
        self.n_out = len(Agent.actionlist)
        self.n_hidden = n_hidden
        self.rawFunction = NeuralNet.FcNN3(self.n_input, self.n_hidden, self.n_out)
        if self.gpu >= 0:
            self.q_func_raw.to_gpu()

    def update(self):
        pass

    def function2field(self, Agent, xlen, ylen):
        field = np.zeros([xlen + 1, ylen + 1])
        xabs = np.pi      # theta
        yabs = 10.0      # omega
        y = np.arange(-yabs, yabs + yabs / ylen, 2 * yabs / ylen)        
        for i in range(0, xlen + 1):
            x = np.ones(ylen + 1) * i * (2.0 * xabs / xlen) - xabs
            sets = np.append([x], [y], axis=0).transpose()
            sets = np.array([Agent.state2grid([ix, iy]) for (ix, iy) in sets])
            field[i] = np.max(self.__call__(sets.astype(np.float32)), axis=0)
        return field


    def drawField(self, Agent, xlen, ylen, epoch, middlename):
        F = self.function2field(Agent, xlen, ylen)
        plt.imshow(F, interpolation='none' )
        plt.ylabel("theta")
        plt.xlabel("omega")
        if epoch == 0:
            print(epoch+1, "draw first figure")
            plt.title("Estimation of Q value in each place at first")
            plt.savefig("DQN_first_" + middlename + ".pdf")
        else:
            # print(epoch+1, "draw latest figure")
            plt.title("Estimation of Q value in each place at latest")
            plt.savefig("DQN_latest_" + middlename + ".pdf")
        plt.close("all")



