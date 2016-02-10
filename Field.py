import numpy as np

class Field(object):
    """
    Field object
    """
    size = np.array([10,20])

    def __init__(self, destination, current=None):
        if current:
            self.crt = current
        else:
            self.crt = self.setinitialposition
        self.dst = destination


    def setinitialposition(self):
        x = np.random.rand(2)
        return x * self.size


    def visualize(self):
        """
        TO DO:
        WRITE FUNCTION TO DRAW THE GOAL AND CURRENT STATUS.

        USE MATPLOTLIB OR PROCESSING FOR PYTHON?
        """
        pass