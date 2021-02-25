import torch
import numpy as np
from matplotlib import pylab as plt
import random
from collections import deque
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Linear, Module
import gym

class FCQSA(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation_fc=F.relu):
        super(FCQSA,self).__init__()
        self.activation_fc = activation_fc
        
        self.input_layer = nn.Linear(input_dim + output_dim, hidden_dims[0])        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],1)
        
        device = 'cpu'
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
          
    def _format(self, state, action):
        s, a = state, action
        if not isinstance(s, torch.Tensor):
              s = torch.tensor(s, device=self.device, dtype=torch.float32)
              s = s.unsqueeze(0)
        if not isinstance(a, torch.Tensor):
              a = torch.tensor(a, device=self.device, dtype=torch.float32)
              a = a.unsqueeze(0)     
        return s, a
          
    def forward(self, state, action):
        s, a = self._format(state, action)
        s = torch.cat((s,a), dim=1)
        s =self.activation_fc(self.input_layer(s))
        for hidden_layer in self.hidden_layers:
            s = self.activation_fc(hidden_layer(s))
        return self.output_layer(s)
        
    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards= torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals
   
class FCGP(nn.Module):
    def __init__(self,
                 input_dim,
                 action_bounds,
                 log_std_min=-20,
                 log_std_max=2,
                 hidden_dims=(32,32),
                 activation_fc=F.relu,
                 entropy_lr=0.001):
        super(FCGP,self).__init__()
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        self.output_layer_mean = nn.Linear(hidden_dims[-1],len(self.env_min))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1],len(self.env_min))

        device = 'cpu'
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
        self.env_min = self._format(self.env_min)
        self.env_max = self._format(self.env_max)
        
        self.nn_min = torch.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = torch.tanh(torch.Tensor([float('inf')])).to(self.device)
        
        self.rescale_fn = lambda x: (x-self.nn_min)*(self.env_max-self.env_min)/(self.nn_max-self.nn_min)+self.env_min
        
        self.target_entropy = -np.prod(self.env_max.shape)
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.logalpha], lr=entropy_lr)
        
    def _format(self, state):
        s = state
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float32)
        return s
        
    def forward(self, state):
        s = self._format(state)
        s =self.activation_fc(self.input_layer(s))
        for hidden_layer in self.hidden_layers:
               s = self.activation_fc(hidden_layer(s))
        mean = self.output_layer_mean(s)
        log_std = torch.clamp(self.output_layer_log_std(s), self.log_std_min, self.log_std_max)
        return mean, log_std
 
    def full_pass(self, state, eps = 1e-6):
        mean, log_std = self.forward(state)
        
        pi_s = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale_fn(tanh_action)
        #log_prob = pi_s.log_prob(pre_tanh_action) - (torch.log((self.env_max-self.env_min)/(self.nn_max-self.nn_min)+(1-tanh_action.pow(2)).clamp(0,1)+eps))#.sum(dim=1, keepdim=True)
        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log((1-tanh_action.pow(2)).clamp(0,1)+eps)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, self.rescale_fn(torch.tanh(mean))
    
    def _update_exploration_ratio(self, greedy_action, action_taken):
        env_min, env_max = self.env_min.cpu().numpy(), self.env_max.cpu().numpy()
        self.exploration_ratio = np.mean(abs(greedy_action - action_taken)/(env_max-env_min))
       
    def _get_actions(self, state):
        mean, log_std = self.forward(state)
        action = self.rescale_fn(torch.tanh(torch.distributions.Normal(mean, log_std.exp()).sample()))
        greedy_action = self.rescale_fn(torch.tanh(mean))
        random_action = np.random.uniform(low=self.env_min.cpu().numpy(), high=self.env_max.cpu().numpy())
    
        action_shape = self.env_max.cpu().numpy().shape
        action = action.detach().cpu().numpy().reshape(action_shape)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(action_shape)
        random_action = random_action.reshape(action_shape)
        
        return action, greedy_action, random_action
        
    def select_random_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, random_action)
        return random_action
    
    def select_greedy_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, greedy_action)
        return greedy_action
    
    def select_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, action)
        return action

class ReplayBuffer():
    def __init__(self,
                 max_size=10000,
                 batch_size=64):
        
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.max_size = max_size
        
    def store(self, sample):
        self.buffer.append(sample)
        
    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
            
        minibatch = random.sample(self.buffer, batch_size)        
        state1_batch = np.array([s1 for (s1,a,r,s2,d) in minibatch])
        action_batch = np.array([a for (s1,a,r,s2,d) in minibatch])
        reward_batch = np.array([r for (s1,a,r,s2,d) in minibatch])
        state2_batch = np.array([s2 for (s1,a,r,s2,d) in minibatch])
        done_batch = np.array([d for (s1,a,r,s2,d) in minibatch])
        
        experiences = state1_batch, \
                      action_batch, \
                      np.vstack(reward_batch), \
                      state2_batch, \
                      np.vstack(done_batch)
        return experiences
    
    def __len__(self):
        return len(self.buffer) 

class SAC():
    def __init__(self,
                 replay_buffer_fn,
                 policy_model_fn,
                 policy_max_grad_norm,
                 policy_optimizer_fn,
                 policy_optimizer_lr,
                 value_model_fn,
                 value_max_grad_norm,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 n_warmup_batches,
                 update_value_target_every_steps,
                 update_policy_target_every_steps,
                 train_policy_every_steps,
                 number_of_networks,
                 tau,
                 alpha_R):
        self.replay_buffer_fn = replay_buffer_fn
        
        self.policy_model_fn = policy_model_fn
        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        
        self.value_model_fn = value_model_fn
        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        
        self.n_warmup_batches = n_warmup_batches
        self.update_value_target_every_steps = update_value_target_every_steps
        self.update_policy_target_every_steps = update_policy_target_every_steps
        self.train_policy_every_steps = train_policy_every_steps
        
        self.number_of_networks = number_of_networks
        
        self.tau = tau
        self.alpha_R = alpha_R
         
    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        current_actions, logpi_s, _ = self.target_policy_model.full_pass(states)
        
        # Optimize alpha parameter
        target_alpha = (self.target_policy_model.target_entropy + logpi_s).detach()
        alpha_loss = - (self.target_policy_model.logalpha * target_alpha).mean()
        
        self.target_policy_model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.target_policy_model.alpha_optimizer.step()
        alpha = self.target_policy_model.logalpha.exp()
        
        # Value function optimization
        with torch.no_grad():
            future_actions, log_pi_fa, _ = self.target_policy_model.full_pass(next_states)
            
            q_sa_p = self.target_networks[0](next_states, future_actions)
            for n in range(1,self.number_of_networks):
                q_sa_p = torch.min(q_sa_p, self.target_networks[n](next_states, future_actions))
            
            td_targets = (rewards - self.mean_reward + self.gamma*(q_sa_p-alpha*log_pi_fa)*(1-is_terminals))
            
            # Updated the mean reward
            self.mean_reward +=self.alpha_R*(td_targets - self.online_networks[0](states, actions)).mean()

        for n in range(self.number_of_networks):            
            q_sa = self.online_networks[n](states, actions)
            value_loss = (td_targets-q_sa).pow(2).mul(0.5).mean() 
            self.value_optimizers[n].zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_networks[n].parameters(), self.value_max_grad_norm)
            self.value_optimizers[n].step()
  
        # Optimize policy model
        if np.sum(self.episode_timestep)%self.train_policy_every_steps == 0:
            current_actions, logpi_s, _ = self.online_policy_model.full_pass(states)
            q_sa_min = self.online_networks[0](states, current_actions)
            for n in range(1,self.number_of_networks):
                q_sa_min = torch.min(q_sa_min,self.online_networks[n](states, current_actions))
                    
            policy_loss = (alpha*logpi_s-q_sa_min).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), self.policy_max_grad_norm)
            self.policy_optimizer.step()
          
    def interaction_step(self, state, env):

        if len(self.replay_buffer) < self.min_samples:
            action = self.online_policy_model.select_random_action(state)
        else:
            action = self.online_policy_model.select_action(state)

        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = state, action, reward, new_state, float(is_failure)
        
        self.replay_buffer.store(experience)
        self.episode_reward[-1]+=reward
        self.episode_timestep[-1]+=1
        return new_state, is_terminal
    
    def update_value_networks(self, tau=None):
        tau = self.tau if tau is None else tau
     
        for n in range(self.number_of_networks):
            for target, online in zip(self.target_networks[n].parameters(), self.online_networks[n].parameters()):
                target.data.copy_((1.0-tau)*target.data + tau*online.data)
    
    def update_policy_network(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_policy_model.parameters(), self.online_policy_model.parameters()):
            target.data.copy_((1.0-tau)*target.data + tau*online.data)
         
    def train(self, env_name, seed, gamma, max_episodes, goal_mean_100_reward):
        self.seed = seed; self.gamma = gamma; self.mean_reward = 0.
        
        env = gym.make(env_name); env.seed(self.seed)
        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)
        
        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        self.episode_reward = []
        self.episode_timestep = []
        
        self.online_networks = []
        self.target_networks = []
        self.value_optimizers = []
        for n in range(self.number_of_networks):
            self.online_networks.append(self.value_model_fn(nS, nA))
            self.target_networks.append(self.value_model_fn(nS,nA))
            self.value_optimizers.append(self.value_optimizer_fn(self.online_networks[-1], self.value_optimizer_lr))
        self.update_value_networks(tau=1.0)
        
        self.online_policy_model = self.policy_model_fn(nS, action_bounds)
        self.target_policy_model = self.policy_model_fn(nS, action_bounds)
        self.update_policy_network(tau=1.0)

        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, self.policy_optimizer_lr)
        
        self.replay_buffer = self.replay_buffer_fn()
        
        self.min_samples = self.replay_buffer.batch_size*self.n_warmup_batches
        
        for episode in range(1, max_episodes+1):
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0)
            
            while not is_terminal:
                state, is_terminal = self.interaction_step(state, env)
                
                if len(self.replay_buffer)>self.min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_networks[0].load(experiences)
                    
                    self.optimize_model(experiences)
                                     
                if np.sum(self.episode_timestep)%self.update_value_target_every_steps == 0:
                    self.update_value_networks()
                    
                if np.sum(self.episode_timestep)%self.update_policy_target_every_steps == 0:
                    self.update_policy_network()
                    
            mean_100_reward = np.mean(self.episode_reward[-100:])
            if mean_100_reward>=goal_mean_100_reward:
                print("Training completed. Average reward is ", mean_100_reward)
                break
            
            if episode%1==0:
                print("Episode: ", episode, "Reward=", self.episode_reward[-1])
                
        env.close(); del env
        return self.episode_reward
 
N = 5
SEEDS=range(11,12)
rewards = [[],[],[],[],[]]
    
for seed in SEEDS:
    env_settings = {
        'env_name': 'Pendulum-v0',#'HalfCheetahBulletEnv-v0',
        'gamma': 1,
        'max_episodes': 10000,
        'goal_mean_100_reward': -150}
        
    policy_model_fn = lambda nS, bounds: FCGP(nS, bounds, hidden_dims=(256, 256))
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0005
    
    value_model_fn = lambda nS, nA: FCQSA(nS, nA, hidden_dims=(256,256))
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0007
    
    replay_buffer_fn = lambda: ReplayBuffer(max_size=1000000, batch_size=256)
    n_warmup_batches = 10
    update_value_target_every_steps = 1
    update_policy_target_every_steps = 2
    train_policy_every_steps = 2
    policy_noise_ratio = 0.1
    policy_noise_clip_ratio = 0.5
    tau = 0.005
    alpha_R = 0.0001
    
    env_name, gamma, max_episodes, goal_mean_100_reward = env_settings.values()
    
    for n in range(1,N+1):
        number_of_networks = 2**n
    
        agent = SAC(replay_buffer_fn,
                         policy_model_fn,
                         policy_max_grad_norm,
                         policy_optimizer_fn,
                         policy_optimizer_lr,
                         value_model_fn,
                         value_max_grad_norm,
                         value_optimizer_fn,
                         value_optimizer_lr,
                         n_warmup_batches,
                         update_value_target_every_steps,
                         update_policy_target_every_steps,
                         train_policy_every_steps,
                         number_of_networks,
                         tau,
                         alpha_R)
        
        rewards[n-1].append(agent.train(env_name, seed, gamma, max_episodes, goal_mean_100_reward))

#plt.plot([np.mean(reward[k:k+10]) for k in range(len(reward)-10)])   
