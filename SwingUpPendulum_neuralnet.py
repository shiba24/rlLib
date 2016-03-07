import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import Qfunction
import six
# import copy

from SwingUpPendulum import Agent
import chainer
from chainer import optimizers
from chainer import serializers

import chainer
import chainer.functions as F
import chainer.links as L


class Pendulum(Agent):
    """
    Pendulum agent.
    """
    def __init__(self, q_func_raw_target=None, memorysize=50, stepsizeparameter=0.9):
        self.memorysize = memorysize

        self.stepsizeparameter = stepsizeparameter
        self.continueflag = True
        self.successflag = False
        self.memory_act = np.array([])

        self.actionlist = (self.plus, self.minus, self.throw)
        self.initializeState()

        if q_func_raw_target is None:
            self.initializeQ()
        else:
            self.q_func_raw = q_func_raw_target.copy()
            self.q_func_raw_target = q_func_raw_target.copy()
            # print(self.q_func_raw, self.q_func_raw_target)

    ### Override ###
    def initializeState(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.omega = np.random.uniform(-5, 5)
        self.state = self.state2grid([self.theta, self.omega])
        self.memory_state = np.array([self.state])

    def initializeQ(self):
        self.q_func_raw = Qfunction.MLP(2, 50, len(self.actionlist))
        self.q_func_raw_target = self.q_func_raw.copy()

    def updateState(self):
        while self.theta < -np.pi:
            self.theta += 2.0 * np.pi
        while self.theta > np.pi:
            self.theta -= 2.0 * np.pi
        self.state = self.state2grid([self.theta, self.omega])

    def state2grid(self, state):
        while state[0] < -np.pi:
            state[0] += 2.0 * np.pi
        while state[0] > np.pi:
            state[0] -= 2.0 * np.pi

        if np.abs(state[0]) < 0.1:
            state[0] = 0.0
        if np.abs(state[1]) < 0.1:
            state[0] = 0.0
        return state

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

    def throw(self):
        ntheta = self.theta + 0.02 * self.omega   # 0.02 sec
        self.omega = self.omega + 0.02*(-0.01*self.omega + 9.8*np.sin(self.theta))
        self.theta = ntheta
        self.updateState()

    def getReward(self):
        reward = np.cos(self.state[0]) # - np.abs(self.state[1])
        return reward
        

    def q_func(self, inputs):
        if len(np.array(inputs).shape) == 1:
            n = 1
        else:
            n = len(inputs)
        return self.q_func_raw.predict(chainer.Variable(np.array(inputs).astype(np.float32).reshape(n, 2))).data.reshape(len(self.actionlist), n)


    def q_func_target(self, inputs):
        if len(np.array(inputs).shape) == 1:
            n = 1
        else:
            n = len(inputs)
        return self.q_func_raw_target.predict(chainer.Variable(np.array(inputs).astype(np.float32).reshape(n, 2))).data.reshape(len(self.actionlist), n)

    def takeAction(self, epsilon, gamma):
        # Get action index due to Strategy
        action_index = self.epsilongreedy(epsilon, self.q_func(self.state))
        # Taking action
        self.actionlist[action_index]()
        self.memory_state = np.append(self.memory_state, [self.state], axis=0)
        self.memory_act = np.append(self.memory_act, action_index)


    def drawField(self, xlen, ylen):
        field = np.zeros([xlen, ylen])
        xabs = np.pi      # theta
        yabs = 5.0      # omega
        y = np.arange(-yabs, yabs, 2 * yabs / ylen)
        
        for i in range(0, xlen):
            x = np.ones(ylen) * i * (2.0 * xabs / xlen) - xabs
            field[i] = np.max(self.q_func(np.append([x], [y], axis=0).transpose().astype(np.float32)), axis=0)
        return field


    def endcheck(self):
        if np.all(self.state == self.state2grid([0.0, 0.0])):
            print("Success!!")
            self.continueflag = False
            self.successflag = True
        if self.theta <- 4.0 * np.pi or self.theta > 4.0 * np.pi:
            self.continueflag = False
        if len(self.memory_state) > 300:
            self.continueflag = False



if __name__ == "__main__":
    # D = makeDatasets()
    # print(D)
    # Qfunc = Qfunction.MLP(2, 50, 2)
    # Qfunc_target = Qfunction.MLP(2, 50, 2)
 
    xp = np

    M = 5
    T = 100

    epsilon = 0.3   
    gamma = 0.99
    memory = 10000
    batchsize = 100


    n_epoch = 100
    f = False

    D = np.zeros([memory, 6]).astype(np.float32)
    y = np.zeros(memory).astype(np.float32)
    cnt = 0

    for i in tqdm(range(0, n_epoch)):
        if i == 0:
            A = Pendulum()
            optimizer = optimizers.MomentumSGD()
            optimizer.setup(A.q_func_raw)
        else:
            A = Pendulum(q_func_raw_target=Q_hat)

        ti = 0
        # print(A.q_func_raw.l1.W.data)

        while A.continueflag:
            A.takeAction(epsilon, gamma)
            # A.updateState()
            reward = A.getReward()
            e = np.append(np.append(A.memory_state[-2], np.array([A.memory_act[-1], reward])), A.memory_state[-1])
            A.endcheck()
            while cnt >= memory:
                print("memory updated")
                f = True
                cnt -= memory

            D[cnt] = e
            if A.successflag == True:
                y[cnt] = reward
            else:
                y[cnt] = reward + gamma * np.max(A.q_func_target(A.memory_state[-1]))
            # print(y[cnt])
            # print(e)
            cnt += 1
            ti += 1

            if f:
                perm = np.random.permutation(memory)
                # ind = 0
                for ind in six.moves.range(0, memory, batchsize):
                    x = chainer.Variable(xp.asarray(D[perm[ind:ind + batchsize], 0:2]))
                    t = chainer.Variable(xp.asarray(y[perm[ind:ind + batchsize]]))

                    # Pass the loss function (Classifier defines it) and its arguments
                    # print(t.data)
                    optimizer.update(A.q_func_raw, x, t, D[perm[ind:ind + batchsize], 2].astype(np.int32))

                    print(A.q_func_raw.loss.data)

                    # if ti % 100 == 0:
                    #     A.q_func_raw_target = A.q_func_raw.copy()
                    # print(A.q_func_raw_target, A.q_func_raw)
                # print(A.q_func_raw.l1.W.data)
                A.continueflag = False

        if i == 0:
            print(i+1, "draw first figure")
            plt.imshow(A.drawField(20, 40), interpolation='none' )
            plt.title("Estimation of Q value in each place at first")
            plt.ylabel("theta")
            plt.xlabel("omega")
            plt.savefig("DQN_1.pdf")
            plt.close("all")
        elif (i + 1) % 10.0 == 0.0:
            print(i+1, "draw latest figure")
            plt.imshow(A.drawField(20, 40), interpolation='none' )
            plt.title("Estimation of Q value in each place at latest")
            plt.ylabel("theta")
            plt.xlabel("omega")
            plt.savefig("DQN_latest.pdf")
            plt.close("all")

        Q_hat = A.q_func_raw#.copy()

    
    plt.imshow(A.drawField(20, 40), interpolation='none' )
    plt.title("Estimation of Q value in each place at last")
    plt.xlabel("omega")
    plt.ylabel("theta")
    plt.savefig("DQN_last.pdf")
    plt.close("all")



    # for epoch in range(0, M):
    #     A = Pendulum()

    #     for t in range(0, T):
    #         A.takeAction(epsilon, gamma)

    #         A.updateState()
    #         reward = A.getReward()
    #         A.updateQ(reward, gamma)





"""


# def training(memory=1000, n_epoch=500, epsilon=0.1, gamma=0.99):
#     D = np.zeros([memory, 6])
#     y = np.zeros(memory)
#     cnt = 0
#     batchsize = 100

#     for i in tqdm(range(0, n_epoch)):
#         if i == 0:
#             A = Pendulum()
#         else:
#             A = Pendulum(q_func_raw=Q)

#         while A.continueflag:
#             A.takeAction(epsilon, gamma)
#             A.updateState()
#             A.endcheck()
#             reward = A.getReward()
#             e = np.append(np.append(A.memory_state[-2], np.array([A.memory_act[-1], reward])), A.memory_state[-1])
#             while cnt >= memory:
#                 cnt -= memory
#             D[cnt] = e
#             if A.continueflag == False:
#                 y[cnt] = reward
#             else:
#                 y[cnt] = reward + gamma * np.max(self.q_func_target(self.memory_state[-1]))

#             cnt += 1

#             perm = np.random.permutation(memory)
#             minibatch = D[perm[0:batchsize]]


#         Q = A.q_func_raw
#     return D


"""

