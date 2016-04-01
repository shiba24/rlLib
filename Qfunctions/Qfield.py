import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from BaseClass import QfunctionBase


class Qfield(QfunctionBase):
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
