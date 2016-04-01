import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class QfunctionBase(object):
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
        field = np.zeros([self.discretize, self.discretize])
        field_min = np.zeros([self.discretize, self.discretize])
        deltax = (staterange[0, 1] - staterange[0, 0]) / self.discretize
        deltay = (staterange[1, 1] - staterange[1, 0]) / self.discretize
        for i in range(0, self.discretize):
            for j in range(0, self.discretize):
                xy = np.array([staterange[0, 0] + deltax * (i + 0.5), staterange[1, 0] + deltay * (j + 0.5)])
                field[i, j] = np.max(self.__call__(xy.astype(np.float32)))
                field_min[i, j] = np.min(self.__call__(xy.astype(np.float32)))
        print(np.max(field), np.max(field_min), np.min(field), np.min(field_min))
        return field

    def drawField(self, Agent, savefilename):
        F = self.function2field(Agent.staterange)
        plt.figure(figsize=(12, 9))
        ax = plt.axes()
        xtickl = np.linspace(Agent.staterange[0, 0], Agent.staterange[0, 1], self.discretize + 1)
        ytickl = np.linspace(Agent.staterange[1, 0], Agent.staterange[1, 1], self.discretize + 1)
        sns.heatmap(F, ax=ax, xticklabels=xtickl, yticklabels=ytickl) #, vmin=-0.1, vmax=0.1)
        locs, labels = sns.plt.xticks()
        sns.plt.setp(labels, rotation=90)
        ax.set_xlabel(Agent.ylabel)
        ax.set_ylabel(Agent.xlabel)
        ax.set_title("Max of estimated Q in each state")
        sns.plt.savefig(savefilename)
        sns.plt.close("all")

    def plotOutputHistory(self, savefigname="history.png"):
        plt.plot(self.rawFunction.history)
        plt.xlabel("N of actions")
        plt.ylabel("Q value (mean)")
        plt.title("Mean Q-function output of each action")
        plt.savefig(savefigname)
        plt.close("all")
