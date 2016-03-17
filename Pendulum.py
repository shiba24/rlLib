import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import six
import argparse

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda

import chainer
import chainer.functions as F
import chainer.links as L

from sklearn.preprocessing import StandardScaler

import Qfunction
from Qfunction import DQN
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


def makeInitialDatasets(datasize, Agent, Qfunc, epsilon, gamma):
    print("making first datasets")
    cnt = 0
    n_row = len(Agent.state) * 2 + 2
    Data = np.zeros([datasize, n_row]).astype(np.float32)
    y = np.zeros(datasize).astype(np.float32)
    notFull = True
    while notFull:
        Agent.initializeState()
        while Agent.continueflag:
            Agent.takeAction(Qfunc, epsilon, gamma)
            reward = Agent.getReward()
            Data[cnt] = np.append(np.append(Agent.memory_state[-2], np.array([Agent.memory_act[-1], reward])),
                                  Agent.memory_state[-1])
            Agent.endcheck()

            if Agent.successflag == True:
                y[cnt] = reward
            else:
                y[cnt] = reward + gamma * np.max(Qfunc(Agent.memory_state[-1]))
            cnt += 1
            # print(cnt)
            if cnt >= len(Data):
                notFull = False
                break
    return Data, y
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN example')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--memorysize', '-m', default=10000, type=int,
                        help='Memory size to remember')
    args = parser.parse_args()

    Sx = StandardScaler() 
    Sy = StandardScaler()

    if args.gpu < 0:
        xp = np 
    else:
        xp = cuda.cupy
        cuda.get_device(args.gpu).use()

    epsilon = 0.15
    gamma = 0.99
    memory = args.memorysize if args.gpu < 0 else 100000
    batchsize = 32  
    n_epoch = 200 if args.gpu < 0 else 30000
    name = "cpu" if args.gpu < 0 else "gpu"


    Agent = Pendulum()
    Q = DQN()
    Qhat = DQN()

    Q.initialize(Agent)
    Qhat.initialize(Agent)

    optimizer = optimizers.MomentumSGD()

    D, y = makeInitialDatasets(memory, Agent, Q, epsilon, gamma)

    cnt = 0
    for epoch in tqdm(range(0, n_epoch)):
        if epoch > 0:
            Agent.initializeState()
            Qhat.rawFunction = Q.rawFunction.copy()

        optimizer.setup(Q.rawFunction)

        while Agent.continueflag:
            Agent.takeAction(Q, epsilon, gamma)
            reward = Agent.getReward()
            if cnt >= memory:
                print("memory updated")
                cnt -= memory
            D[cnt] = np.append(np.append(Agent.memory_state[-2], np.array([Agent.memory_act[-1], reward])),
                               Agent.memory_state[-1])
            Agent.endcheck()
            if Agent.successflag == True:
                y[cnt] = reward
            else:
                y[cnt] = reward + gamma * np.max(Qhat(Agent.memory_state[-1]))
            cnt += 1

            # data scaling
            X = Sx.fit_transform(D[:, 0:2])
            y_scaled = Sy.fit_transform(y.reshape(-1, 1)).reshape(-1, )

            # training
            sum_loss, ind = 0, 0
            perm = np.random.permutation(memory)
            # for ind in six.moves.range(0, memory, batchsize):
            x = chainer.Variable(xp.asarray(X[perm[ind:ind + batchsize], 0:2]))
            t = chainer.Variable(xp.asarray(y_scaled[perm[ind:ind + batchsize]]))
            indexes = chainer.Variable(xp.asarray(D[perm[ind:ind + batchsize], 2].astype(np.int32)))

            optimizer.update(Q.rawFunction, x, t, indexes)
            sum_loss += float(Q.rawFunction.loss.data) * len(t.data)

            if cnt % 100 == 0:
                Qhat.rawFunction = Q.rawFunction.copy()

        if epoch % 10.0 == 0.0:
            Q.drawField(Agent, 20, 40, epoch, name)

    




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






