import numpy as np
import pickle
import six


def makeInitialDatasets(datasize, Agent, Qfunc, epsilon=1.0, gamma=0.99):
    print("making first datasets...")
    cnt = 0
    n_row = len(Agent.state) * 2 + 2
    Data = np.zeros([datasize, n_row]).astype(np.float32)
    y = np.zeros(datasize).astype(np.float32)
    notFull = True
    while notFull:
        Agent.initializeState()
        # Agent.endcheck(volatile=False)
        while Agent.continueflag:
            Agent.volatile = False
            Agent.takeAction(Qfunc, epsilon)
            reward = Agent.getReward()
            Data[cnt] = np.append(np.append(Agent.memory_state[-2],
                                            np.array([Agent.memory_act[-1], reward])),
                                            Agent.memory_state[-1])

            # if len(Agent.memory_state) > 2:
                # Data[cnt] = np.append(np.append(np.append(Agent.memory_state[-3], Agent.memory_state[-2]),
                #                                 np.array([Agent.memory_act[-1], reward])),
                #                                 Agent.memory_state[-1])
            # else:
            #     Data[cnt] = np.append(np.append(np.append(np.zeros(2), Agent.memory_state[-2]),
            #                                     np.array([Agent.memory_act[-1], reward])),
            #                                     Agent.memory_state[-1])

            Agent.endcheck()
            if Agent.continueflag:
                y[cnt] = reward + gamma * np.max(Qfunc(Agent.memory_state[-1]))
            else:
                y[cnt] = reward
            # print(cnt, Agent.memory_state[-1], Qfunc(Agent.memory_state[-1]), y[cnt])

            cnt += 1
            if cnt >= len(Data):
                notFull = False
                break
    return Data, y


def saveData(X, y, savename="data.pkl"):
    print("saving datasets......")
    Data = {}
    Data["X"] = X
    Data["y"] = y
    with open(savename, 'wb') as output:
        six.moves.cPickle.dump(Data, output, -1)


def loadData(filename):
    print("loading datastes......")
    with open(filename, 'rb') as pk:
        data = six.moves.cPickle.load(pk)
    return data["X"], data["y"]

