import numpy as np
from Agent import Agent


class Searchway(Agent):
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
        super(Searchway, self).__init__(memorysize, gamma, stepsizeparameter, forcedinfield=forcedinfield, volatile=volatile)
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



class Pendulum(Agent):
    """
    Pendulum agent.
    The child class of Agent class should include:
        * Set action functions and actionlist = (plus, minus) as tuples
            - Each action function should return the state if it is selected.
        * getReward function
            - defines and returns reward
    """
    onestep = 5.0
    staterange = np.array([[-np.pi, np.pi], [-5.0, 5.0]])   # [[xmin, xmax], [ymin, ymax]]
    goalrange = np.array([[-0.1, 0.1], [-10.0, 10.0]])      # [[xmin, xmax], [ymin, ymax]]
    xlabel = "theta"
    ylabel = "omega"

    def __init__(self, memorysize=50, gamma=0.9, stepsizeparameter=0.9, forcedinfield=False):
        super(Pendulum, self).__init__(memorysize, gamma, stepsizeparameter, forcedinfield)
        self.actionlist = (self.plus, self.minus, self.throw)

    """In this task, there is a special eq about state."""
    def scaleState(self, state):
        while state[0] < - np.pi:
            state[0] += 2.0 * np.pi
        while state[0] > np.pi:
            state[0] -= 2.0 * np.pi
        return state

    """ Actions: Return state array """
    def plus(self):
        if not (self.state[1] > self.staterange[1, 1] - self.onestep and self.forcedinfield):
            ntheta = self.state[0] + 0.02 * self.state[1]   # 0.02 sec
            omega = self.state[1] + 0.02 * (-0.01 * self.state[1] + 9.8 * np.sin(self.state[0]) + 5.0)
            return self.scaleState([ntheta, omega])
        else:
            self.throw()

    def minus(self):
        if not (self.state[1] < self.staterange[1, 0] + self.onestep and self.forcedinfield):
            ntheta = self.state[0] + 0.02 * self.state[1]   # 0.02 sec
            omega = self.state[1] + 0.02 * (-0.01 * self.state[1] + 9.8 * np.sin(self.state[0]) - 5.0)
            return self.scaleState([ntheta, omega])
        else:
            self.throw()

    def throw(self):
        ntheta = self.state[0] + 0.02 * self.state[1]   # 0.02 sec
        omega = self.state[1] + 0.02 * (-0.01 * self.state[1] + 9.8 * np.sin(self.state[0]))
        return self.scaleState([ntheta, omega])

    """Reward function"""
    def getReward(self):
        reward = np.cos(self.state[0])  #- np.abs(self.state[1]) / 5.0
        # reward = 10 - np.abs(self.state[1])
        return reward



"""
if self.state[1] >= self.staterange[1, 0] + self.onestep and self.forcedinfield:
    self.state[1] -= self.onestep
elif not self.forcedinfield:
    self.state[1] -= self.onestep
return self.state
"""
