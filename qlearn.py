import numpy as np
from tqdm import tqdm
import six
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Qfunction import Qfield
import Task
resultdir = "./result/"

parser = argparse.ArgumentParser(description='Q learning example')
parser.add_argument('--task', '-t', choices=('Pendulum', 'Searchway'),
                    default='Searchway', help='task name')
parser.add_argument('--memorysize', '-m', default=100, type=int,
                    help='The memory size for action')
parser.add_argument('--nepoch', '-n', default=5000, type=int,
                    help='The number of epoch')
parser.add_argument('--epsilon', '-e', default=0.1, type=float,
                    help='Last epsilon value for epsilon-greedy policy')
parser.add_argument('--gamma', '-g', default=0.99, type=float,
                    help='Time constance gamma for update Q')
parser.add_argument('--alpha', '-a', default=0.9, type=float,
                    help='Stepsize parameter')
args = parser.parse_args()

# parameter settings
final_epsilon = args.epsilon
gamma = args.gamma
memory = args.memorysize
n_epoch = args.nepoch
stepsizeparameter = args.alpha
num_actions = []

# Agent and Qfunction settings
if args.task == "Pendulum":
    Agent = Task.Pendulum(memorysize=memory, gamma=gamma, stepsizeparameter=stepsizeparameter)
    name = "pen"
    xlab = "omega"
    ylab = "theta"
elif args.task == "Searchway":
    Agent = Task.Searchway(memorysize=memory, gamma=gamma, stepsizeparameter=stepsizeparameter)
    name = "way"
    xlab = "x"
    ylab = "y"

Q = Qfield(discretize=20)
Q.initialize(Agent)

# Initial datesets settings
for epoch in tqdm(range(0, n_epoch)):
    Agent.initializeState()

    # drawing q-function figure
    if epoch == 0:
        print(epoch+1, "draw first figure")
        # print(Agent.staterange)
        Q.drawField(Agent.staterange, 
                    resultdir + "q_qf_" + name + "_first.pdf", xlabel=xlab, ylabel=ylab)
    elif epoch % 10.0 == 0.0:
        # print(Agent.staterange)
        Q.drawField(Agent.staterange, 
                    resultdir + "q_qf_" + name + "_latest.pdf", xlabel=xlab, ylabel=ylab)

    epsilon = np.max(1.0 - epoch * 0.9 * 10 / n_epoch, final_epsilon)
    if epoch > n_epoch * 9 / 10:
        Agent.forcedinfield = True

    while Agent.continueflag:
        # Action of agents
        Agent.volatile = False
        Agent.takeAction(Q, epsilon)
        reward = Agent.getReward()
        Agent.endcheck()

        if Agent.continueflag:
            Q.update(Agent, reward)

    num_actions.append(len(Agent.memory_act))

    # drawing result figure
    if epoch == 0.0:
        Agent.drawField(resultdir + "q_track_" + name + "_first_q.pdf")
    elif epoch % 10.0 == 0.0:
        Agent.drawField(resultdir + "q_track_" + name + "_latest_q.pdf")
