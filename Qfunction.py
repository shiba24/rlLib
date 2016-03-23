import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import NeuralNet
import chainer


class Qfunction(object):
    """
    Qfunction class for generalized tasks of q-learning.
    This class should be argument of Agent.takeAction(qFunction, epsilon, gamma)
        qFunction can be both np.ndarray, or function object.
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def function2field(self, staterange):
        # field = np.zeros([self.discretize + 1, self.discretize + 1])
        field = np.zeros([self.discretize, self.discretize])
        field_min = np.zeros([self.discretize, self.discretize])
        field_right = np.zeros([self.discretize, self.discretize])
        deltax = (staterange[0, 1] - staterange[0, 0]) / self.discretize
        deltay = (staterange[1, 1] - staterange[1, 0]) / self.discretize
        for i in range(0, self.discretize):
            for j in range(0, self.discretize):
                xy = np.array([staterange[0, 0] + deltax * (i + 0.5), staterange[1, 0] + deltay * (j + 0.5)])
                field[i, j] = np.max(self.__call__(xy.astype(np.float32)))
                field_right[i, j] = self.__call__(xy.astype(np.float32))[0, 0]
                field_min[i, j] = np.min(self.__call__(xy.astype(np.float32)))
                # print(xy, self.__call__(xy.astype(np.float32)), field[i, j])
                # f[i, j] = xy[0]
        # print(f)
        print(np.max(field), np.max(field_min), np.min(field), np.min(field_min))
        # field = field - np.min(field)
        return field, field_right

    def drawField(self, staterange, savefilename, xlabel="omega", ylabel="theta"):
        F, Fr = self.function2field(staterange)
        plt.figure(figsize=(12, 9))
        ax = plt.axes()
        xtickl = np.linspace(staterange[0, 0], staterange[0, 1], self.discretize + 1)
        ytickl = np.linspace(staterange[1, 0], staterange[1, 1], self.discretize + 1)
        sns.heatmap(F, ax=ax, xticklabels=xtickl, yticklabels=ytickl) #, vmin=-0.1, vmax=0.1)
        locs, labels = sns.plt.xticks()
        sns.plt.setp(labels, rotation=90)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Estimation of Q value in each place")
        # plt.imshow(F, interpolation='none')
        sns.plt.savefig(savefilename)
        sns.plt.close("all")

        plt.figure(figsize=(12, 9))
        ax = plt.axes()
        sns.heatmap(Fr, ax=ax, xticklabels=xtickl, yticklabels=ytickl) #, vmin=-0.1, vmax=0.1)
        locs, labels = sns.plt.xticks()
        sns.plt.setp(labels, rotation=90)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Estimation of Q value in each place of action 1")
        sns.plt.savefig("right.png")
        sns.plt.close("all")


    def plotOutputHistory(self, savefigname="history.png"):
        plt.plot(self.rawFunction.history)
        plt.xlabel("N of actions")
        plt.ylabel("Q value (mean)")
        plt.title("Mean Q-function output of each action")
        plt.savefig(savefigname)
        plt.close("all")


class Qfield(Qfunction):
    def __init__(self, discretize=10):
        self.discretize = discretize

    def __call__(self, inputs):
        # index = list(np.array(inputs).astype(np.int32))
        index = self.state2index(inputs)
        output = self.rawFunction[tuple(index)]
        # print(output)
        return output

    def state2index(self, inputs):
        return list(np.array((inputs - self.staterange[:, 0]) / self.delta).astype(np.int32))

    def initialize(self, Agent):
        self.staterange = Agent.staterange.copy()
        self.delta =  np.diff(self.staterange, axis=1).reshape(1, -1)[0] / self.discretize
        statedim = len(Agent.state)
        self.rawFunction = np.random.random(np.append(np.ones(statedim) * self.discretize,
                                                      len(Agent.actionlist)).astype(np.int32)) * 0.1 + 0.5

    # Q_new = reward + gamma max_a' Q(s', a')
    def update(self, Agent, reward):
        # print(Agent.memory_state[-1])
        index_st2 = self.state2index(Agent.memory_state[-1])
        target = reward + Agent.gamma * np.max(self.rawFunction[tuple(index_st2)])
        index_st1 = self.state2index(Agent.memory_state[-2])
        # print(index_st1)
        diff = target - self.rawFunction[tuple(np.append(index_st1, Agent.memory_act[-1]))]
        self.rawFunction[tuple(np.append(index_st1, Agent.memory_act[-1]))] += Agent.stepsizeparameter * diff

    def function2field(self, staterange):
        return np.max(self.rawFunction, axis=2)


class DQN(Qfunction):
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

    def update(self):
        pass

    # def function2field(self, staterange, xlen, ylen):
    #     field = np.zeros([xlen + 1, ylen + 1])
    #     deltax = (staterange[0, 1] - staterange[0, 0]) / xlen
    #     deltay = (staterange[1, 1] - staterange[1, 0]) / ylen
    #     for i in range(0, xlen + 1):
    #         for j in range(0, ylen + 1):
    #             xy = np.array([staterange[0, 0] + deltax * i, staterange[1, 0] + deltay * j])
    #             field[i, j] = np.max(self.__call__(xy.astype(np.float32)))
    #     print(np.max(field), np.min(field))
    #     # field = field - np.min(field)
    #     return field

    # def drawField(self, staterange, xlen, ylen, savefilename, xlabel="omega", ylabel="theta"):
    #     F = self.function2field(staterange, xlen, ylen)
    #     plt.imshow(F, interpolation='none')
    #     plt.ylabel(ylabel)
    #     plt.xlabel(xlabel)
    #     plt.title("Estimation of Q value in each place")
    #     plt.savefig(savefilename)
    #     plt.close("all")

    # def function2field(self, Agent, xlim, ylim, xlen, ylen):
    #     field = np.zeros([xlen + 1, ylen + 1])
    #     deltax = (xlim[1] - xlim[0]) / xlen
    #     deltay = (ylim[1] - ylim[0]) / ylen
    #     y = np.arange(ylim[0], ylim[1] + deltay, deltay)
    #     for i in range(0, xlen + 1):
    #         x = np.ones(ylen + 1) * (xlim[0] + deltax * i)
    #         pairs = np.append([x], [y], axis=0).transpose()
    #         sets = np.array([Agent.state2grid([ix, iy]) for (ix, iy) in pairs])
    #         field[i] = np.max(self.__call__(sets.astype(np.float32)), axis=0)
    #     return field
