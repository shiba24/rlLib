import numpy as np
from Agent import Agent

class Pendulum(Agent):
    """
    Pendulum agent.
    The child class of Agent class should include:
        * Set specified state = [theta, omega]
            - override InitializeState if necessary
        * Set action functions and actionlist = (plus, minus) as tuples
            - Each action function should return the state if it is selected.
        * getReward function
            - defines and returns reward
        * Endcheck function due to specific state
    """
    def __init__(self, memorysize=50, stepsizeparameter=0.9):
        super(Pendulum, self).__init__(memorysize, stepsizeparameter)
        self.actionlist = (self.plus, self.minus)
        self.initializeState()

    """Specify state and initializeState"""
    def initializeState(self):
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-5, 5)
        self.state = self.state2grid([theta, omega])
        self.memory_state = np.array([self.state])
        self.continueflag = True
        self.successflag = False

    def state2grid(self, state):
        while state[0] < - np.pi:
            state[0] += 2.0 * np.pi
        while state[0] > np.pi:
            state[0] -= 2.0 * np.pi
        return state

    """ Actions: Return state array """
    def plus(self):
        ntheta = self.state[0] + 0.02 * self.state[1]   # 0.02 sec
        self.state[1] = self.state[1] + 0.02*(-0.01*self.state[1] + 9.8*np.sin(self.state[0]) + 5.0)
        self.state[0] = ntheta
        return self.state2grid([self.state[0], self.state[1]])

    def minus(self):
        ntheta = self.state[0] + 0.02 * self.state[1]   # 0.02 sec
        self.state[1] = self.state[1] + 0.02*(-0.01*self.state[1] + 9.8*np.sin(self.state[0]) - 5.0)
        self.state[0] = ntheta
        return self.state2grid([self.state[0], self.state[1]])

    def throw(self):
        ntheta = self.state[0] + 0.02 * self.state[1]   # 0.02 sec
        self.state[1] = self.state[1] + 0.02*(-0.01*self.state[1] + 9.8*np.sin(self.state[0]))
        self.state[0] = ntheta
        return self.state2grid([self.state[0], self.state[1]])

    """Reward function"""
    def getReward(self):
        # reward = np.cos(self.state[0])  #- np.abs(self.state[1]) / 5.0
        reward = 10 - np.abs(self.state[1])
        return reward

    """Endcheck function"""
    def endcheck(self):
        if np.all(self.state == self.state2grid([0.0, 0.0])):
            print("Success!!")
            self.continueflag = False
            self.successflag = True
        # if self.state[0] <- 4.0 * np.pi or self.state[0] > 4.0 * np.pi:
        #     self.continueflag = False
        if len(self.memory_state) > 30:
            self.continueflag = False




"""


    def initialize(self, n_input, n_hidden, n_out):
        self.q_func_raw = NeuralNet.FcNN3(n_input, n_hidden, n_out)
        self.q_func_raw_target = self.q_func_raw.copy()
        if self.gpu >= 0:
            self.q_func_raw.to_gpu()
            self.q_func_raw_target.to_gpu()


    def q_func(self, inputs):
        if len(xp.array(inputs).shape) == 1:
            n = 1
        else:
            n = len(inputs)
        if self.gpu < 0:
            return self.q_func_raw.predict(chainer.Variable(np.array(inputs).astype(np.float32).reshape(n, 2))).data.transpose()
        else:
            return xp.asnumpy(self.q_func_raw.predict(chainer.Variable(xp.asarray(np.array(inputs).astype(np.float32).reshape(n, 2)))).data.transpose())

    def q_func_target(self, inputs):
        if len(xp.array(inputs).shape) == 1:
            n = 1
        else:
            n = len(inputs)
        if self.gpu < 0:
            return self.q_func_raw_target.predict(chainer.Variable(np.array(inputs).astype(np.float32).reshape(n, 2))).data.transpose()
        else:
            return xp.asnumpy(self.q_func_raw_target.predict(chainer.Variable(xp.asarray(np.array(inputs).astype(np.float32).reshape(n, 2)))).data.transpose())



"""






