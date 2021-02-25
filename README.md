# Some-analysis-of-the-Soft-Actor-Critic-algorithm

## Problem setting
We aim to perform a series of empirical experiments with the Soft Actor-Critic (SAC) algorithm developed in [Haarnoja et al.(2018)](https://arxiv.org/abs/1801.01290). In the SAC algorithm's standard implementation, two neural networks are approximating Q-value, called online networks. The networks are independent, i. e. they have separate layers and independent optimizers. Besides, there are two copies of these networks, called target networks, which provide Q-values for calculating the action-value targets. The motivation for using two Q-functions is to mitigate the positive bias in the policy improvement step. This bias is introduced by overestimation of action-values, which can adversely influence the learning performance of a value-based algorithm, see [Hasselt (2010)](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).

The intuition dictates that adding more Q-functions may improve the learning performance further. We want to study how the original algorithm's performance will change if the number of neural networks, which approximate the value function, increase. For this study's purpose, we use a standard 'Pendulum-v0' environment provided by [Gym OpenAI](https://gym.openai.com/envs/Pendulum-v0/).

We extend the implementation of the SAC algorithm presented by [Miguel Morales (@mimoralea)](https://github.com/mimoralea) in several ways. Besides the number of networks for Q-value approximation, other changes include:
* The average reward setting is used. This setting is typical for continuing problems like the 'Pendulum-v0'environment without episode boundaries. It should replace discounting setting as suggested by [Sutton and Barto (2018)](http://www.incompleteideas.net/book/RLbook2020.pdf) (see Chapter 10.3 and 13.6) under the function approximation.
* Target policy network is added for better stability. 
* Target policy smoothing based on Polyak averaging and delaying of update is incorporated.
* Replay buffer is based on ''deque'' data structure.
* The order of the networks' parameters update is changed to avoind numerical issues.


## Numerical experiments
In our numerical experiments, we run several training sessions of the SAC agents with 2, 4, 8, 16, and 32 independent networks approximation Q-value. For each session, we repeat the experiment across ten random seeds. The training ends when the agent achieves a moving average of -150 across the previous 100 episodes. 

![Figure 1](https://github.com/NikolayGudkov/Some-analysis-of-the-Soft-Actor-Critic-algorithm/blob/main/SAC_plus_1.png)

From the figure above, we see for all ten random seeds, there is a minor, if any, the difference in the SAC algorithm's performance with varying the number of Q-networks for solving the "Pendulum-v0" environment.

For a single seed, the algorithm's performance for a different number of Q-networks is very similar during the initial training phase and indistinguishable during the final stage, as seen from the figure below.

![Figure 2](https://github.com/NikolayGudkov/Some-analysis-of-the-Soft-Actor-Critic-algorithm/blob/main/SAC_plus_2.png)

The minor difference in the performance of various specifications can be explained by the fact that all networks are trained using the same batch of experiences collected from one replay buffer (unlike in the parallelized algorithms like A3C, A2C, PPO, etc., in which experiences are collected in parallel). Therefore, the estimated parameters for all networks converge to similar values. The Q-functions are very close to each other, which agrees with the results shown in the second figure above.

## Conclusion
Although the performance across different SAC specifications with a varying number of networks is similar, the computational burden of having many networks is much severe. Therefore, there is no computational benefit from adding more than two neural networks to approximate Q-function.

## Future research
Perform similar experiments in more complex environments like 'HopperBulletEnv-v0' and 'HalfCheetahBulletEnv-v0'
