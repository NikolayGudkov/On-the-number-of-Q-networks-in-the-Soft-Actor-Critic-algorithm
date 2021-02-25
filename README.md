# Some-analysis-of-the-Soft-Actor-Critic-algorithm

We aim to perform a series of empirical experiments with the Soft Actor-Critic (SAC) algorithm developed in [Haarnoja et al.(2018)](https://arxiv.org/abs/1801.01290). In the standard implementation of the SAC algorithm, there are two neural networks approximating Q-value, called online networks. The networks are independent, that is they have independent layers and independent optimizers. Besides, there are two copies of these networks, called target networks, which provide Q-values used for calculation of the action-value targets. The motivation for using two Q-funtions is to mitigate the positive bias in the policy improvement step. This bias is introduced by overestimation of action-values, which can adversely influence the learning performance of a value-based algorithm, see [Hasselt (2010)](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).

The intuition dictates that adding more Q-functions may improve the learning performance further. We want to study how the performace of the original algorithm will change if the number of neural networks, which approximate the value function, increase. For the purpose of this study, we use standard 'Pendulum-v0' environment provided by [Gym OpenAI](https://gym.openai.com/envs/Pendulum-v0/).

We extend the implementation of the SAC algorithm presented by [Miguel Morales (@mimoralea)](https://github.com/mimoralea) in several ways. Besides the number of networks approximation Q-value, other changes include:
..* The average reward setting is used. This setting is typical for continuing problems like the 'Pendulum-v0'environment without episode boundaries. It should replace discounting setting as suggest by [Sutton and Barto (2018)](http://www.incompleteideas.net/book/RLbook2020.pdf) (see Chapter 10.3 and 13.6) under the function approximation.
..* Target policy network is added for better stability. Target smoothing based on Polyak averaging is incorporated.
..* Replay buffer is based on ''deque'' data structure.
..* The order of network paramter update is changed.
