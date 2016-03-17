import numpy as np
import types
import Qfunction


class Policy(object):
    """
    Policy sets for agent to choice:    {random, greedy, epsilongreedy}
    """
    def __init__(self):
        pass

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


class Agent(Policy):
    """
    Agent class for generalized tasks of q-learning.
    function sets about:
        state (initialize, update)
        takeAction, due to Policy.
    """
    def __init__(self, memorysize=50, stepsizeparameter=0.9):
        self.memorysize = memorysize
        self.stepsizeparameter = stepsizeparameter
        self.continueflag = True
        self.successflag = False
        self.memory_act = np.array([])
        self.memory_state = np.array([])


    def initializeState(self, startstate):
        self.state = startstate
        self.memory_state = np.array([self.state])

        
    def updateState(self, actionFunction, *args):
        self.state = actionFunction(*args)


    def takeAction(self, qFunction, epsilon, gamma):
        # Get action index due to Policy
        if isinstance(qFunction, np.ndarray):
            action_index = self.epsilongreedy(epsilon, qFunction[tuple(self.state)])
        elif isinstance(qFunction, types.FunctionType) or isinstance(qFunction, Qfunction.Qfunction):
            action_index = self.epsilongreedy(epsilon, qFunction(self.state))
        else:
            action_index = self.random()
        # action and update
        self.updateState(self.actionlist[action_index])
        # memorize
        self.memory_state = np.append(self.memory_state, [self.state], axis=0)
        self.memory_act = np.append(self.memory_act, action_index)




