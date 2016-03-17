import numpy as np
from tqdm import tqdm
import six
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from sklearn.preprocessing import StandardScaler

from Qfunction import DQN
from Searchway import Searchway
from utils import makeInitialDatasets
resultdir = "./result/"


parser = argparse.ArgumentParser(description='DQN example - searchway')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--memorysize', '-m', default=10000, type=int,
                    help='Memory size to remember')
parser.add_argument('--batchsize', '-b', default=100, type=int,
                    help='Minibatch size')
parser.add_argument('--nepoch', '-n', default=3000, type=int,
                    help='The number of epoch')
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
batchsize = args.batchsize
n_epoch = args.nepoch if args.gpu < 0 else 30000
name = "way_cpu" if args.gpu < 0 else "way_gpu"

# Agent and Qfunction settings
Agent = Searchway()
Q = DQN(gpu=args.gpu)
Q.initialize(Agent)
Qhat = DQN(gpu=args.gpu)
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
        if Agent.successflag:
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

        if cnt % 30 == 0:
            Qhat.rawFunction = Q.rawFunction.copy()

    # drawing result figure
    if epoch % 10.0 == 0.0:
        Q.drawField(Agent, [0.0, 10.0], [0.0, 10.0], 40, 40, epoch, name, xlabel="x", ylabel="y")

    if epoch == 0:
        plt.plot(Agent.memory_state.T[0], Agent.memory_state.T[1])
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.title("State Track at first")
        plt.savefig(resultdir + "state_track_first_dqn.pdf")
        plt.close()


plt.plot(Agent.memory_state.T[0], Agent.memory_state.T[1])
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title("State Track at last")
plt.savefig(resultdir + "state_track_last_dqn.pdf")
plt.close()
