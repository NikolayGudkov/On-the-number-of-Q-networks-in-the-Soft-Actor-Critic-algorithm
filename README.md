# Some-analysis-of-the-Soft-Actor-Critic-algorithm

We aim to perform a series of empirical experiments with the Soft Actor-Critic (SAC) algorithm developed in [Haarnoja et al.(2018)](https://arxiv.org/abs/1801.01290). In the standard implementation of the SAC algorithm, there are two neural networks approximating Q-value, called online networks. The networks are independent, that is they have independent layers and independent optimizers. Besides, there are two copies of these networks, called target networks, which provide Q-values used for calculation of the action-value targets. The motivation for using two Q-funtions is to address the positive bias introduced by overestimation of action-values, which can adversely influence the learning performance of a value-based algorithm, see [Hasselt (2010)](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).

The intuition dictates that adding more Q-functions We want to study how the performace of the original algorithm will change if the number of neural networks, which approximate the value function, increase. For this purpose, we use standard 'Pendulum-v0' environment provided by [Gym OpenAI](https://gym.openai.com/envs/Pendulum-v0/).
