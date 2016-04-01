import numpy as np
from BaseClass import AgentBase


class PendulumSolver(AgentBase):
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
        super(PendulumSolver, self).__init__(memorysize, gamma, stepsizeparameter, forcedinfield)
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
