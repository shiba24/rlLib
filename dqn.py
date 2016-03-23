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

from Qfunction import DQN
import Task
import utils
resultdir = "./result"

parser = argparse.ArgumentParser(description='DQN example - searchway')
parser.add_argument('--task', '-t', choices=('Pendulum', 'Searchway'),
                    default='Searchway', help='task name')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--memorysize', '-m', default=10000, type=int,
                    help='Memory size to remember')
parser.add_argument('--epsilon', '-e', default=0.1, type=float,
                    help='Last epsilon value for epsilon-greedy policy')
parser.add_argument('--alpha', '-a', default=0.9, type=float,
                    help='Stepsize parameter')
parser.add_argument('--batchsize', '-b', default=100, type=int,
                    help='Minibatch size')
parser.add_argument('--nepoch', '-n', default=3000, type=int,
                    help='The number of epoch')
args = parser.parse_args()

# parameter settings
if args.gpu < 0:
    xp = np
else:
    xp = cuda.cupy
    cuda.get_device(args.gpu).use()


gamma = 0.9
final_epsilon = args.epsilon
stepsizeparameter = args.alpha
memory = args.memorysize            # if args.gpu < 0 else 100000
batchsize = args.batchsize
n_epoch = args.nepoch               # if args.gpu < 0 else 30000
num_actions = []

# Agent and Qfunction settings
if args.task == "Pendulum":
    Agent = Task.Pendulum(memorysize=memory / 10, stepsizeparameter=stepsizeparameter)
    name = "pen_cpu" if args.gpu < 0 else "pen_gpu"
    xlab = "omega"
    ylab = "theta"
elif args.task == "Searchway":
    Agent = Task.Searchway(memorysize=memory / 10, stepsizeparameter=stepsizeparameter)
    name = "way_cpu" if args.gpu < 0 else "way_gpu"
    xlab = "y"
    ylab = "x"


Q = DQN(discretize=20, gpu=args.gpu)
Q.initialize(Agent, n_hidden=50)
Qhat = DQN(discretize=20, gpu=args.gpu)
Qhat.initialize(Agent, n_hidden=50)

# Initial datesets settings
Agent.forcedinfield = False
Agent.volatile = True

D, y = utils.makeInitialDatasets(memory, Agent, Q, epsilon=1.0, gamma=gamma)
utils.saveData(D, y)

optimizer = optimizers.MomentumSGD(lr=0.00025, momentum=0.95)
optimizer.setup(Q.rawFunction)
# chainer.optimizer.WeightDecay(0.00005)
# chainer.optimizer.GradientClipping(1.0)

cnt = 0
C = 0
# training loop
for epoch in tqdm(range(0, n_epoch)):
    Agent.initializeState()
    # Agent.endcheck(volatile=False)
    # Qhat.rawFunction = Q.rawFunction.copy()

    # drawing q-function figure
    if epoch == 0:
        print(epoch + 1, "draw first figure")
        Q.drawField(Agent.staterange, 
                    resultdir + "dqn_qf_" + name + "_first.pdf", xlabel=xlab, ylabel=ylab)
    elif epoch % 10.0 == 0.0:
        Q.drawField(Agent.staterange, 
                    resultdir + "dqn_qf_" + name + "_latest.pdf", xlabel=xlab, ylabel=ylab)

    epsilon = np.max(1.0 - epoch * 0.9 * 4 / n_epoch, final_epsilon)

    while Agent.continueflag:
        # Action of agents
        Agent.takeAction(Q, epsilon)
        reward = Agent.getReward()
        if cnt >= memory:
            print("memory updated")
            cnt -= memory
        Agent.endcheck()

        D[cnt] = np.append(np.append(Agent.memory_state[-2],
                                     np.array([Agent.memory_act[-1], reward])),
                                     Agent.memory_state[-1])

        if Agent.continueflag:
            y[cnt] = reward + gamma * np.max(Qhat(Agent.memory_state[-1]))
        else:
            y[cnt] = reward

        cnt += 1
        C += 1

        # data scaling and standardization
        X = D[:, 0:2] # / 10.0 - 0.5
        y_scaled = y # / 10.0

        # training, random sampling of the size of minibatch-size
        sum_loss = 0
        perm = np.random.permutation(memory)
        ind = 0
        # ind = cnt + batchsize if cnt + batchsize < memory else cnt + batchsize - memory
        # for ind in six.moves.range(0, memory, batchsize):     # if you want to train with all the datasets
        x = chainer.Variable(xp.asarray(X[perm[ind:ind + batchsize]]))
        t = chainer.Variable(xp.asarray(y_scaled[perm[ind:ind + batchsize]]))
        indexes = chainer.Variable(xp.asarray(D[perm[ind:ind + batchsize], 2].astype(np.int32)))

        optimizer.update(Q.rawFunction, x, t, indexes)
        sum_loss += float(Q.rawFunction.loss.data) * len(t.data)

        if C % 1000 == 0:
            print("#################### Q function copied! ####################", C)
            Qhat.rawFunction = Q.rawFunction.copy()

        if len(Agent.memory_act) > 100 and len(Agent.memory_act) % 50 == 0:
            Agent.drawField(resultdir + "dqn_track_" + name + "_latest.png")

    num_actions.append(len(Agent.memory_act))
    Q.plotOutputHistory(resultdir + "history_q.png")

    plt.plot(num_actions)
    plt.title("Searching way to destinaton result")
    plt.xlabel("n_epoch")
    plt.ylabel("n_of actions")
    plt.savefig(resultdir + "dqn_n_actions_" + name + ".pdf")
    plt.close("all")
    