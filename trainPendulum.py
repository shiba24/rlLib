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

from Qfunction import DQN
from Pendulum import Pendulum


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
 


parser = argparse.ArgumentParser(description='DQN example')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--memorysize', '-m', default=10000, type=int,
                    help='Memory size to remember')
args = parser.parse_args()

Sx = StandardScaler() 
Sy = StandardScaler()

# parameter settings
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

# Agent and Qfunction settings
Agent = Pendulum()
Q = DQN()
Q.initialize(Agent)
Qhat = DQN()
Qhat.initialize(Agent)

# Initial datesets settings
D, y = makeInitialDatasets(memory, Agent, Q, epsilon, gamma)

optimizer = optimizers.MomentumSGD()

cnt = 0
for epoch in tqdm(range(0, n_epoch)):
    if epoch > 0:
        Agent.initializeState()
        Qhat.rawFunction = Q.rawFunction.copy()

    optimizer.setup(Q.rawFunction)

    while Agent.continueflag:
        # Action of agents
        Agent.takeAction(Q, epsilon, gamma)
        reward = Agent.getReward()
        if cnt >= memory:
            print("memory updated")
            cnt -= memory
        Agent.endcheck()

        # New data acquisition for D and y
        D[cnt] = np.append(np.append(Agent.memory_state[-2], np.array([Agent.memory_act[-1], reward])),
                           Agent.memory_state[-1])
        if Agent.successflag == True:
            y[cnt] = reward
        else:
            y[cnt] = reward + gamma * np.max(Qhat(Agent.memory_state[-1]))
        cnt += 1

        # data scaling and standardization
        X = Sx.fit_transform(D[:, 0:2])
        y_scaled = Sy.fit_transform(y.reshape(-1, 1)).reshape(-1, )

        # training, random sampling of the size of minibatch-size
        sum_loss, ind = 0, 0
        perm = np.random.permutation(memory)
        # for ind in six.moves.range(0, memory, batchsize):     # if you want to train with all the datasets
        x = chainer.Variable(xp.asarray(X[perm[ind:ind + batchsize], 0:2]))
        t = chainer.Variable(xp.asarray(y_scaled[perm[ind:ind + batchsize]]))
        indexes = chainer.Variable(xp.asarray(D[perm[ind:ind + batchsize], 2].astype(np.int32)))

        optimizer.update(Q.rawFunction, x, t, indexes)
        sum_loss += float(Q.rawFunction.loss.data) * len(t.data)

        if cnt % 100 == 0:
            Qhat.rawFunction = Q.rawFunction.copy()

    # drawing result figure
    if epoch % 10.0 == 0.0:
        Q.drawField(Agent, 20, 40, epoch, name)
    

