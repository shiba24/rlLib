Reinforcement learning
======

Introduction (Q learning) to application (Deep Q Network) of reinforcement learning.

## Requirements
### Q-learning
[tqdm](https://github.com/noamraph/tqdm)

[matplotlib v1.5.1](http://matplotlib.org/)

[numpy v1.10.1](http://www.numpy.org/)

### DQN

[chainer v1.5.1](http://chainer.org/)

[scikit-learn](http://scikit-learn.org/stable/)

[Other packages](https://github.com/pfnet/chainer#requirements) for chainer.


## Usage
### Q-learning

Learning the shortest way to the destination of given field, using q-learning algorithm with epsilon-greedy strategy.
```
$ python SearchWay.py
```

Learning how to [swimg up the pendulum](https://www.youtube.com/watch?v=YLAWnYAsai8), using q-learning algorithm with epsilon-greedy strategy.
```
$ python SwingUpPendulum.py
```


### DQN
Learning how to [swimg up the pendulum](https://www.youtube.com/watch?v=YLAWnYAsai8), using [deep-q network](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html) algorithm originated from nature.
```
$ python SwingUpPendulum_neuralnet.py
```
If you want to use GPU, add option of ```-g 0```.


## Author

If you have troubles or questions, please contact [shiba24](https://github.com/shiba24).

March, 2016

