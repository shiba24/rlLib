import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import types
from reinforcement_learning.Qfunctions.BaseClass import QfunctionBase


class PolicyBase(object):
    """
    Policy sets for agent to choice:    {random, greedy, epsilongreedy}
    """
    def __init__(self):
        pass

    def random(self):
        return np.random.randint(len(self.actionlist))

    def greedy(self, q_list):
        if len(self.actionlist) != len(q_list):
            raise AssertionError("The length of actionlist and must be the same as the length of q_list!")
        return np.argmax(q_list)

    def epsilongreedy(self, epsilon, q_list):
        if np.random.rand(1) < epsilon:
            return self.random()
        else:
            return self.greedy(q_list)


class AgentBase(PolicyBase):
    """
    Agent class for generalized tasks of q-learning.
    function sets about:
        state (initialize, update)
        takeAction, due to Policy.
    """
    def __init__(self, memorysize=100, gamma=0.9, stepsizeparameter=0.9, forcedinfield=True, volatile=True):
        self.memorysize = memorysize
        self.stepsizeparameter = stepsizeparameter
        self.gamma = gamma
        self.forcedinfield = forcedinfield
        self.volatile = volatile
        self.xlabel = "x"
        self.ylabel = "y"
        self.initializeState()

    def updateState(self, actionFunction, *args):
        self.state = actionFunction(*args)

    def initializeState(self):
        # self.state = np.random.rand(len(self.staterange)) * (self.staterange[:, 1] - self.staterange[:, 0]) + self.staterange[:, 0]
        self.state = np.random.rand(len(self.staterange)) - 0.5
        while self.goalcheck():
            self.state = np.random.rand(len(self.staterange)) - 0.5
            # self.state = np.random.rand(len(self.staterange)) * (self.staterange[:, 1] - self.staterange[:, 0]) + self.staterange[:, 0]
        self.memory_state = np.array([self.state])
        self.memory_act = np.array([])
        self.continueflag = True
        self.successflag = False

    def scaleState(self, state):
        state = (state - np.mean(self.staterange, axis=1)) / np.diff(self.staterange, axis=1).reshape(1,-1)[0]
        return state

    def scaleStateInv(self, state):
        state = state * np.diff(self.staterange, axis=1).reshape(1,-1)[0] + np.mean(self.staterange, axis=1)
        return state

    def goalcheck(self):
        if np.all([self.goalrange[i, 0] <= self.state[i] <= self.goalrange[i, 1] for i in range(0, len(self.goalrange))]):
            return True
        else:
            return False
        
    def fieldoutcheck(self):
        if np.all([self.staterange[i, 0] < self.state[i] < self.staterange[i, 1] for i in range(0, len(self.staterange))]):
            return False
        else:
            return True

    def takeAction(self, qFunction, epsilon):
        # Get action index due to Policy
        if isinstance(qFunction, np.ndarray):
            action_index = self.epsilongreedy(epsilon, qFunction[tuple(self.state)])
        elif isinstance(qFunction, types.FunctionType) or isinstance(qFunction, QfunctionBase):
            action_index = self.epsilongreedy(epsilon, qFunction(self.state))
        else:
            action_index = self.random()
        # action and update
        self.updateState(self.actionlist[action_index])
        # memorize
        self.memory_state = np.append(self.memory_state, [self.state], axis=0)
        self.memory_act = np.append(self.memory_act, action_index)

    def drawField(self, savefigname):
        plt.plot(self.memory_state.T[0], self.memory_state.T[1])
        plt.xlim(tuple(self.staterange[0]))
        plt.ylim(tuple(self.staterange[1]))
        plt.title("State Track")
        plt.savefig(savefigname)
        plt.close()

    def getReward(self):
        raise NotImplementedError

    def endcheck(self):
        if self.goalcheck():
            self.continueflag = False
            self.successflag = True
            print("Goal!   ", len(self.memory_act), len(self.memory_state), self.memory_state[0], self.state)
        elif self.fieldoutcheck():
            self.continueflag = False
            self.successflag = False
            print("Give up!", len(self.memory_act), len(self.memory_state), self.memory_state[0], self.state)
        elif len(self.memory_state) > self.memorysize and self.volatile:
            self.continueflag = False
            self.successflag = False
            print("Give up!", len(self.memory_act), len(self.memory_state), self.memory_state[0], self.state)
