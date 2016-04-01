import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
from tqdm import tqdm
import six
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainer import cuda

from reinforcement_learning.Qfunctions.DQN import DQN
from reinforcement_learning.Tasks import Pendulum
from reinforcement_learning.Tasks import Searchway
from reinforcement_learning.Utils import utils

resultdir = "./data/"


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
    Agent = Pendulum.PendulumSolver(memorysize=memory / 10, stepsizeparameter=stepsizeparameter)
    name = "pen_cpu" if args.gpu < 0 else "pen_gpu"
elif args.task == "Searchway":
    Agent = Searchway.SearchwaySolver(memorysize=memory / 10, stepsizeparameter=stepsizeparameter)
    name = "way_cpu" if args.gpu < 0 else "way_gpu"


Q = DQN(discretize=20, gpu=args.gpu)
Q.initialize(Agent, n_hidden=50)
Q.setupOptimizerSGD(lr=0.00025, momentum=0.95)

Qhat = DQN(discretize=20, gpu=args.gpu)
Qhat.copy(Q)

# Initial datesets settings
Agent.forcedinfield = False
Agent.volatile = True

X, y = utils.makeInitialDatasets(memory, Agent, Q, epsilon=1.0, gamma=gamma)

cnt = 0
C = 0
# training loop
for epoch in tqdm(range(0, n_epoch)):
    Agent.initializeState()

    # drawing q-function figure
    if epoch == 0:
        print(epoch + 1, "draw first figure")
        Q.drawField(Agent, resultdir + "dqn_qf_" + name + "_first.pdf")
    elif epoch % 10.0 == 0.0:
        Q.drawField(Agent, resultdir + "dqn_qf_" + name + "_latest.pdf")

    epsilon = np.max(1.0 - epoch * 0.9 * 4 / n_epoch, final_epsilon)

    while Agent.continueflag:
        # Action of agents
        Agent.takeAction(Q, epsilon)
        reward = Agent.getReward()
        if cnt >= memory:
            print("memory updated")
            cnt -= memory
        Agent.endcheck()

        X[cnt] = np.append(np.append(Agent.memory_state[-2],
                                     np.array([Agent.memory_act[-1], reward])),
                                     Agent.memory_state[-1])

        if Agent.continueflag:
            y[cnt] = reward + gamma * np.max(Qhat(Agent.memory_state[-1]))
        else:
            y[cnt] = reward

        cnt += 1
        C += 1

        # data scaling and standardization
        Q.update(X, y, batchsize)

        if C % 1000 == 0:
            print("#################### Q function copied! ####################", C)
            Qhat.rawFunction = Q.rawFunction.copy()

        if len(Agent.memory_act) > 100 and len(Agent.memory_act) % 50 == 0:
            Agent.drawField(resultdir + "dqn_track_" + name + "_latest.png")

    num_actions.append(len(Agent.memory_act))
    Q.plotOutputHistory(resultdir + "dqn_history_" + name + ".png")

    # plot num of action results
    plt.plot(num_actions)
    plt.title("Searching way to destinaton result")
    plt.xlabel("n_epoch")
    plt.ylabel("n_of actions")
    plt.savefig(resultdir + "dqn_n_actions_" + name + ".png")
    plt.close("all")
    


"""
##### previous version
        # ind = cnt + batchsize if cnt + batchsize < memory else cnt + batchsize - memory
        # for ind in six.moves.range(0, memory, batchsize):     # if you want to train with all the datasets
"""
