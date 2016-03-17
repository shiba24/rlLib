import numpy as np
from Agent import Agent


class Searchway(Agent):
    """
    Searching-way agent.
    The child class of Agent class should include:
        * Set specified state = [theta, omega]
            - override InitializeState if necessary
        * Set action functions and actionlist = (plus, minus) as tuples
            - Each action function should return the state if it is selected.
        * getReward function
            - defines and returns reward
        * Endcheck function due to specific state
    """

    fieldsize = np.array([0, 10])

    def __init__(self, memorysize=50, stepsizeparameter=0.9):
        super(Searchway, self).__init__(memorysize, stepsizeparameter)
        self.actionlist = (self.right, self.left, self.up, self.down)
        self.initializeState()

    """Specify state and initializeState"""
    def initializeState(self):
        # 0: upside-down, 1: rightside-left
        self.state = np.random.rand(2) * self.fieldsize[1]
        self.memory_state = np.array([self.state])
        self.continueflag = True
        self.successflag = False

    def state2grid(self, state):
        return state

    """ Actions: Return state array """
    def right(self):
        if self.state[1] < self.fieldsize[1]:
            self.state[1] += 0.1
        return self.state

    def left(self):
        if self.state[1] > self.fieldsize[1]:
            self.state[1] -= 0.1
        return self.state

    def up(self):
        if self.state[0] < self.fieldsize[0]:
            self.state[0] += 0.1
        return self.state

    def down(self):
        if self.state[0] > self.fieldsize[0]:
            self.state[0] -= 0.1
        return self.state

    """Reward function"""
    def getReward(self):
        if 2.7 < self.state[0] < 3.3 and 6.7 < self.state[1] < 7.3:
            reward = 1
        else:
            reward = -0.5
        return reward

    """Endcheck function"""
    def endcheck(self):
        if 2.7 < self.state[0] < 3.3 and 6.7 < self.state[1] < 7.3:
            self.continueflag = False
            self.successflag = False
        if len(self.memory_state) > 300:
            self.continueflag = False


"""
        # print("Goal! Actions = ", len(A.memory_act))
    # env = Environment(fieldsize, destination)

        # self.q_field[self.destination[0], self.destination[1], :] = np.ones([1, len(self.actionlist)])

        # if len(self.memory_act) > 6:
        #     ind1 = self.memory_pos[-6:-1][0]
        #     diff = self.field[self.memory_pos[-6:-1][0], self.memory_pos[-6:-1][1]] - self.q_field[self.memory_pos[-7:-2][0], self.memory_pos[--7:2][1], self.memory_act[-6:-1]]
        #     self.q_field[self.memory_pos[-7:-2][0], self.memory_pos[-7:-2][1],
        #                  self.memory_act[-6:-1]] += self.stepsizeparameter * diff
        # else:


    # def searchAction(self, gamma):
    #     reward = self.field[self.memory_pos[-1, 0], self.memory_pos[-1, 1]]
    #     target = reward + gamma * np.max(self.q_field[self.memory_pos[-1][0], self.memory_pos[-1][1], :])

"""

