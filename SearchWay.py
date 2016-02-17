import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Environment(object):
    def __init__(self, fieldsize, destination):
        self.fieldsize = fieldsize
        self.positioncheck(destination)
        self.destination = destination
        self.field = np.zeros(self.fieldsize, dtype=np.int32)
        self.field[self.destination[0], self.destination[1]] = 1

    def positioncheck(self, position):
        if type(position) == int or len(position) != 2:
            print("position = ", position)
            raise ValueError("Position should be 1 x 2 array! check the position of the agent.")                        
        if position[0] > self.fieldsize[0] or position[1] > self.fieldsize[1]:
            print("position = ", position)
            raise ValueError("Position expanded fieldsize! check the position of the agent.")

    def __call__(self):
        return self.field


class Agent(Environment):

    q_fieldsize = 10

    def __init__(self, fieldsize, destination, startposition, q_field=None, memorysize=10, stepsizeparameter=0.9):
        Environment.__init__(self, fieldsize, destination)
        self.positioncheck(startposition)
        self.position = startposition
        self.memorysize = memorysize
        self.actionlist = (self.right, self.left, self.up, self.down)
        self.continueflag = True

        self.memory_pos = np.array(self.position)      # 0 = position, 1 =action
        self.memory_act = np.array([])

        self.stepsizeparameter = stepsizeparameter
        if q_field is None:
            self.initializeQ()
        else:
            self.q_field = q_field


    def right(self):
        if self.position[1] < self.fieldsize[1] - 1:
            self.position[1] += 1
    def left(self):
        if self.position[1] > 0:
            self.position[1] -= 1
    def up(self):
        if self.position[0] < self.fieldsize[0] - 1:
            self.position[0] += 1
    def down(self):
        if self.position[0] > 0:
            self.position[0] -= 1


    def takeAction(self, epsilon):
        if np.random.rand(1) < epsilon:
            action_index = np.random.randint(len(self.actionlist))
        else:
            ls = self.q_field[self.position[0], self.position[1], :]
            index_list = np.argwhere(ls == np.amax(ls))
            # action_index = np.argmax(self.q_field[self.position[0], self.position[1], :])
            action_index = index_list[np.random.permutation(len(index_list))[0]][0]

        self.memory_pos = np.c_[self.memory_pos, self.position]
        self.memory_act = np.append(self.memory_act, action_index)
        self.actionlist[action_index]()
        self.updateQ()
        self.endcheck()


    def initializeQ(self):
        self.q_field = np.ones([self.q_fieldsize, self.q_fieldsize, len(self.actionlist)]) * 0.5


    def updateQ(self):
        diff = self.field[self.memory_pos[-1][0], self.memory_pos[-1][1]] - self.q_field[self.memory_pos[-2][0], self.memory_pos[-2][1], self.memory_act[-1]]
        self.q_field[self.memory_pos[-2][0], self.memory_pos[-2][1],
                     self.memory_act[-1]] += self.stepsizeparameter * diff


    def endcheck(self):
        if self.field[self.position[0], self.position[1]] == 1:
            self.continueflag = False



if __name__ == "__main__":
    destination = np.array([2, 2])
    startposition = np.array([6, 6])
    fieldsize = np.array([10, 10])
    n_epoch = 1000
    k = []

    for i in tqdm(range(0, n_epoch)):
        if i == 0:
            A = Agent(fieldsize, destination, startposition)
            plt.imshow(np.mean(A.q_field, axis=2), interpolation='none')
            plt.title("Estimation of Q value in each place at first")
            plt.savefig("weights_first.png")
            plt.close("all")
        else:
            A = Agent(fieldsize, destination, startposition, q_field=Q)

        while A.continueflag:
            A.takeAction(0.05)
        k.append(len(A.memory_act))
        Q = A.q_field


    # plotting results
    plt.imshow(np.mean(A.q_field, axis=2), interpolation='none' )
    plt.title("Estimation of Q value in each place at last")
    plt.savefig("weights_last.png")
    plt.close()

    plt.imshow(A.q_field[:, :, 1], interpolation='none' )
    plt.title("Estimation of Q value moving to the left in each place")
    plt.savefig("weights_last_left.png")
    plt.close()


    plt.plot(k)
    plt.title("Searching way to destinaton result")
    plt.xlabel("n_epoch")
    plt.ylabel("n_of actions")
    plt.savefig("result.png")
    plt.close("all")


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
"""

