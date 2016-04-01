Reinforcement learning
======

Simple library of reinforcement learning (Q learning and Deep Q Network).


## Overview

Q learning is a classical algorithm for reinforcement learning.

Deep Q Network (DQN) is originated from [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html) by Volodymyr Mnih et al.

You can [define your own task](https://github.com/shiba24/reinforcement_learning#define-your-own-task) with little effort and solve it with q-learning or dqn algorithm easily!

## Requirements
### Q-learning
- [tqdm](https://github.com/noamraph/tqdm)

- [matplotlib v1.5.1](http://matplotlib.org/)

- [numpy v1.10.1](http://www.numpy.org/)

- [seaborn](https://stanford.edu/~mwaskom/software/seaborn/)

### DQN

- [chainer v1.5.1 +](http://chainer.org/)

- [Other packages](https://github.com/pfnet/chainer#requirements) for chainer.

## Usage: Solve sample tasks
### Q-learning

Learning the shortest way to the destination of given field, using q-learning algorithm with epsilon-greedy strategy.
```
$ python qlearn.py --task "Searchway"
```

Learning how to [swimg up the pendulum](https://www.youtube.com/watch?v=YLAWnYAsai8), using deep-q network.
```
$ python qlearn.py --task "Pendulum"
```

### DQN
Learning the tasks using Deep Q Network.

```
$ python dqn.py --task "Searchway"
```

```
$ python dqn.py --task "Pendulum"
```


## Usage: Define your own task

Now in prep...


## Author

[shiba24](https://github.com/shiba24), March, 2016

