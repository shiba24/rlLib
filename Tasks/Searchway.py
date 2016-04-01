import numpy as np
from BaseClass import AgentBase


class SearchwaySolver(AgentBase):
    """
    Searching-way agent.
    The child class of Agent class should include:
        * Set action functions and actionlist = (plus, minus) as tuples
            - Each action function should return the state if it is selected.
        * getReward function
            - defines and returns reward
    """

    onestep = 0.1
    # 0: upside-down, 1: rightside-left
    staterange = np.array([[-0.5, 0.5], [-0.5, 0.5]])   # [[xmin, xmax], [ymin, ymax]]
    goalrange = np.array([[-0.3, -0.2], [0.1, 0.2]])      # [[xmin, xmax], [ymin, ymax]]

    def __init__(self, memorysize=5000, gamma=0.9, stepsizeparameter=0.9, forcedinfield=True, volatile=True):
        super(SearchwaySolver, self).__init__(memorysize, gamma, stepsizeparameter, forcedinfield=forcedinfield, volatile=volatile)
        self.actionlist = (self.right, self.left, self.up, self.down)

    """ Actions: Return state array """
    def right(self):
        if not (self.state[1] > self.staterange[1, 1] - self.onestep and self.forcedinfield):
            self.state[1] += self.onestep
        return self.state

    def left(self):
        if not (self.state[1] < self.staterange[1, 0] + self.onestep and self.forcedinfield):
            self.state[1] -= self.onestep
        return self.state

    def up(self):
        if not (self.state[0] > self.staterange[0, 1] - self.onestep and self.forcedinfield):
            self.state[0] += self.onestep
        return self.state

    def down(self):
        if not (self.state[0] < self.staterange[0, 0] + self.onestep and self.forcedinfield):
            self.state[0] -= self.onestep
        return self.state

    """Reward function"""
    def getReward(self):
        if self.goalcheck():
            reward = 1.0
        elif self.fieldoutcheck():
            reward = -1.0
        elif len(self.memory_state) > self.memorysize and self.volatile:
            reward = -1.0
        else:
            reward = -0.0
        return reward
