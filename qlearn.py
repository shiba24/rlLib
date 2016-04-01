import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
from tqdm import tqdm
import six
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforcement_learning.Qfunctions.Qfield import Qfield
from reinforcement_learning.Tasks import Pendulum
from reinforcement_learning.Tasks import Searchway


resultdir = "./data/"

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
    Agent = Pendulum.PendulumSolver(memorysize=memory, gamma=gamma, stepsizeparameter=stepsizeparameter)
    name = "pen"
elif args.task == "Searchway":
    Agent = Searchway.SearchwaySolver(memorysize=memory, gamma=gamma, stepsizeparameter=stepsizeparameter)
    name = "way"

Q = Qfield(discretize=10)
Q.initialize(Agent)

# Initial datesets settings
for epoch in tqdm(range(0, n_epoch)):
    Agent.initializeState()

    # drawing q-function figure
    if epoch == 0:
        print(epoch + 1, "draw first figure")
        Q.drawField(Agent, resultdir + "q_qf_" + name + "_first.pdf")
    elif epoch % 10.0 == 0.0:
        Q.drawField(Agent, resultdir + "q_qf_" + name + "_latest.pdf")

    epsilon = np.max(1.0 - epoch * 0.9 * 10 / n_epoch, final_epsilon)

    while Agent.continueflag:
        # Action of agents
        Agent.volatile = False
        Agent.takeAction(Q, epsilon)
        reward = Agent.getReward()
        Agent.endcheck()

        if Agent.continueflag:
            Q.update(Agent, reward)

        if len(Agent.memory_act) > 100 and len(Agent.memory_act) % 50 == 0:
            Agent.drawField(resultdir + "q_track_" + name + "_latest.png")

    num_actions.append(len(Agent.memory_act))

    # drawing result figure
