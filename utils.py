import numpy as np

def makeInitialDatasets(datasize, Agent, Qfunc, epsilon, gamma):
    print("making first datasets...")
    cnt = 0
    n_row = len(Agent.state) * 2 + 2
    Data = np.zeros([datasize, n_row]).astype(np.float32)
    y = np.zeros(datasize).astype(np.float32)
    notFull = True
    while notFull:
        Agent.initializeState()
        while Agent.continueflag:
            Agent.takeAction(Qfunc, epsilon, gamma)
            reward = Agent.getReward()
            Data[cnt] = np.append(np.append(Agent.memory_state[-2], np.array([Agent.memory_act[-1], reward])),
                                  Agent.memory_state[-1])
            Agent.endcheck()
            if Agent.successflag:
                y[cnt] = reward
            else:
                y[cnt] = reward + gamma * np.max(Qfunc(Agent.memory_state[-1]))
            cnt += 1
            if cnt >= len(Data):
                notFull = False
                break
    return Data, y
