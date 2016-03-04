import sys
import cPickle as pickle
import datetime, math, sys, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# https://homes.cs.washington.edu/~todorov/courses/amath579/reading/Continuous.pdf

class Strategy(object):
    """
    Strategy sets for agent to choice:    {random, greedy, epsilongreedy}
    """
    def __init__(self):
        """ Need to override self.actionlist ! """
        self.actionlist = ()

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


class Agent(Strategy):
    """
    Agent class for generalized tasks of q-learning.
    function sets about:
        state (initialize, update)
        Q-estimation (initialize, update)
        takeAction, due to some Strategy.
    """
    def __init__(self, startstate=None, q_field=None, memorysize=50, stepsizeparameter=0.9):
        self.memorysize = memorysize
        if startstate is not None:
            self.initializeState(startstate)
        self.stepsizeparameter = stepsizeparameter
        self.continueflag = True
        self.memory_act = np.array([])

    def initializeState(self, startstate):
        self.state = startstate
        self.memory_state = np.array([[self.state]])

    def initializeQ(self, size):
        self.q_field = np.random.random(np.append(size, len(self.actionlist))) * 0.1 + 0.5
        
    def takeAction(self, epsilon, gamma):
        # Get action index due to Strategy
        action_index = self.epsilongreedy(epsilon, self.q_field[tuple(self.state)])
        self.memory_state = np.append(self.memory_state, [self.state], axis=0)
        self.memory_act = np.append(self.memory_act, action_index)
        # Taking action
        self.actionlist[action_index]()

    def updateState(self):
        """ Dummy. need to be overridden """
        pass

    # Q_new = reward + gamma max_a' Q(s', a')
    def updateQ(self, reward, gamma):
        target = reward + gamma * np.max(self.q_field[tuple(self.memory_state[-1])])
        diff = target - self.q_field[tuple(np.append(self.memory_state[-2], self.memory_act[-1]))]
        self.q_field[tuple(np.append(self.memory_state[-2], self.memory_act[-1]))] += self.stepsizeparameter * diff


class Pendulum(Agent):
    """
    Pendulum agent.
    """
    def __init__(self, fieldsize=[13, 13], q_field=None, memorysize=50, stepsizeparameter=0.9):
        Agent.__init__(self, q_field=q_field, memorysize=memorysize, stepsizeparameter=stepsizeparameter)

        self.actionlist = (self.plus, self.minus)
        self.initializeState()
        if q_field is None:
            self.initializeQ(fieldsize)
        else:
            self.q_field = q_field

    ### Override ###
    def initializeState(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.omega = np.random.uniform(-4.5, 4.5)
        self.state = self.state2grid([self.theta, self.omega])
        self.memory_state = np.array([self.state])

    def updateState(self):
        self.state = self.state2grid([self.theta, self.omega])

    def state2grid(self, state):
        while state[0] < -np.pi:
            state[0] += 2.0 * np.pi
        while state[0] > np.pi:
            state[0] -= 2.0 * np.pi
        if state[1] >= 4.5:
            state[1] = 4.4
        if state[1] <= -4.5:
            state[1] = -4.4
        return np.array([np.round(state[0] * 2) + 6, np.round(state[1] * 1.3) + 6]).astype(np.int32)      # [theta, omega]

    ### Action list ###
    def plus(self):
        ntheta = self.theta + 0.02 * self.omega   # 0.02 sec
        self.omega = self.omega + 0.02*(-0.01*self.omega + 9.8*np.sin(self.theta) + 5.0)
        self.theta = ntheta
        self.updateState()

    def minus(self):
        ntheta = self.theta + 0.02 * self.omega   # 0.02 sec
        self.omega = self.omega + 0.02*(-0.01*self.omega + 9.8*np.sin(self.theta) - 5.0)
        self.theta = ntheta
        self.updateState()

    def getReward(self):
        reward = np.cos(self.theta)
        return reward

    def endcheck(self):
        if np.all(self.state == self.state2grid([0, 0])):
            print("Success!!")
            self.continueflag = False
        if self.theta <- 4.0 * np.pi or self.theta > 4.0 * np.pi:
            self.continueflag = False
        if len(self.memory_state) > 300:
            self.continueflag = False


if __name__ == "__main__":
    n_epoch = 10000
    num_actions = []
    epsilon = 0.1
    gamma = 0.99

    for i in tqdm(range(0, n_epoch)):
        if i == 0:
            A = Pendulum()
        else:
            A = Pendulum(q_field=Q)

        while A.continueflag:
            A.takeAction(epsilon, gamma)
            A.updateState()
            A.updateQ(A.getReward(), gamma)
            A.endcheck()

        num_actions.append(len(A.memory_act))
        Q = A.q_field

    plt.imshow(np.max(A.q_field, axis=2), interpolation='none' )
    plt.title("Estimation of Q value in each place at last")
    plt.xlabel("theta")
    plt.ylabel("omega")
    plt.savefig("Q.pdf")
    plt.close()

    plt.plot(num_actions)
    plt.title("Searching way to destinaton result")
    plt.xlabel("n_epoch")
    plt.ylabel("n_of actions")
    plt.savefig("result.pdf")
    plt.close("all")




