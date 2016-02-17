#from Field import Field
import numpy as np

from tqdm import tqdm

destination = np.array([2.0, 2.0])
startposition = np.array([8.0, 8.0])


class Environment(object):

    def __init__(self, destination):
        self.destination = destination


    def __call__(self):
        return 環境のマップ





class Agent(object):

    def __init__(self, env_array, startposition):
        self.q_field = env_array
        self.position = startposition

    def sensory(self, Environment):
        環境からセンサーとして入力を受ける


    def takeaction(self):
        estimated!に基づき、epsilon-greedyで行動を選択
        return 行動

    def updateQ(self):
        行動の選択系列に基づき、Q(s,a)を更新


    def endcheck(self):
        if self.position == destination:
            self.endflag = True
        else:



if __name__ "__main__":
    n_epoch = 100000
    t = np.zeros(0)

    env = Environment(destination)

    for i in tqdm(range(0, n_epoch)):
        A = Agent

        if a.endflag:
            np.append(t, Agent.行動回数)


    plot t




"""



pd.DataFrame(np.array(a.as_matrix()), columns=["id", "price"])


def horzconcat(parent, daughter):
    if daughter in parent[""]
















"""