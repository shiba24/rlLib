Reinforcement learning
======

Introduction (Q learning) to application (Deep Q Network) of reinforcement learning.

You can define your own task with little effort and solve it with q-learning or dqn algorithm easily!

## Requirements
### Q-learning
[tqdm](https://github.com/noamraph/tqdm)

[matplotlib v1.5.1](http://matplotlib.org/)

[numpy v1.10.1](http://www.numpy.org/)

[seaborn](https://stanford.edu/~mwaskom/software/seaborn/)

### DQN

[chainer v1.5.1 +](http://chainer.org/)

[Other packages](https://github.com/pfnet/chainer#requirements) for chainer.


## Usage
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


## Define your own task

Now in preparation...


## Author

[shiba24](https://github.com/shiba24), March, 2016

