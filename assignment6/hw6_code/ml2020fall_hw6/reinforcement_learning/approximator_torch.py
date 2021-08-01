from collections import deque
import numpy as np
import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class DQN(nn.Module):
    
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
from collections import namedtuple
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)






class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        # begin answer
        self.head = nn.Linear(state_size, action_size)
        # end answer
        pass

    def forward(self, state):
        qvalues = None
        # begin answer
        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        flatten_state=state.view(state.size(0), -1)
        qvalues=self.head(flatten_state)
        # end answer
        return qvalues


class Approximator:
    '''Approximator for Q-Learning in reinforcement learning.
    
    Note that this class supports for solving problems that provide
    gym.Environment interface.
    '''

    def __init__(self,
                 state_size,
                 action_size,
                 learning_rate=0.001,
                 gamma=0.95,
                 init_epsilon=1.0,
                 epsilon_decay=0.995,
                 min_epsilon=0.01,
                 batch_size=32,
                 memory_pool_size=10000,
                 double_QLearning=False):
        '''Initialize the approximator.

        Args:
            state_size (int): the number of states for this environment. 
            action_size (int): the number of actions for this environment.
            learning_rate (float): the learning rate for training optimzer for approximator.
            gamma (float): the gamma factor for reward decay.
            init_epsilon (float): the initial epsilon probability for exploration.
            epsilon_decay (float): the decay factor each step for epsilon.
            min_epsilon (float): the minimum epsilon in training.
            batch_size (int): the batch size for training, only applicable for experience replay.
            memory_pool_size (int): the maximum size for memory pool for experience replay.
            double_QLearning (bool): whether to use double Q-learning.
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.memory = deque(maxlen=memory_pool_size)
        self.batch_size = batch_size
        self.double_QLearning = double_QLearning
        # save the approximator model in self.model
        self.model = None
        # implement your approximator below
        self.model = Model(self.state_size, self.action_size)
        # begin answer
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # end answer
    
    def add_to_memory(self, state, action, reward, new_state, done):
        """Add the experience to memory pool.

        Args:
            state (int): the current state.
            action (int): the action to take.
            reward (int): the reward corresponding to state and action.
            new_state (int): the new state after taking action.
            done (bool): whether the decision process ends in this state.
        """
        # begin answer
        # if len(self.memory) < self.memory_pool_size:
        experience=(state,action,reward,new_state,done)#也可以搞一个字典
        self.memory.append(experience)
        # end answer
        pass
    
    def take_action(self, state):
        """Determine the action for state according to Q-value and epsilon-greedy strategy.
        
        Args:
            state (int): the current state.

        Returns:
            action (int): the action to take.
        """
        if isinstance(state, np.ndarray):
            state = torch.Tensor(state)
        action = 0
        # begin answer
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            # action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
            action=random.randint(0,self.action_size-1)
        if self.epsilon>self.min_epsilon:
            self.epsilon*=self.epsilon_decay
        # end answer
        return int(action)
    
    def online_training(self, state, action, reward, new_state, done):
        """Train the approximator with a batch.

        Args:
            state (tuple(Tensor)): the current state.
            action (tuple(int)): the action to take.
            reward (tuple(float)): the reward corresponding to state and action.
            new_state (tuple(Tensor)): the new state after taking action.
            done (tuple(bool)): whether the decision process ends in this state.
        """
        states = torch.stack(state)  # BatchSize x StateSize
        next_states = torch.stack(new_state)  # BatchSize x StateSize
        actions = torch.Tensor(action).long()  # BatchSize
        rewards = torch.Tensor(reward)  # BatchSize
        masks = torch.Tensor(done)  # BatchSize. Note that 1 means done

        # begin answer
        state_action_values=self.model(states)
        next_state_values=self.model(next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward
        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        # end answer
        pass
    
    def experience_replay(self):
        """Use experience replay to train the approximator.
        """
        # HINT: You may find `zip` is useful.
        # begin answer
        
        # end answer
        pass
    
    def train(self, env, total_episode):
        """Train the approximator.

        Args:
            env (gym.Environment): the environment that provides gym.Environment interface.
            total_episode (int): the number of episodes to train.
        """
        # save the rewards for each training episode in self.reward_list.
        self.reward_list = []
        # Hint: you need to change the reward returned by env to be -1
        #   if the decision process ends at one step.
        # begin answer
        
        # end answer
        
        all_rewards = 0
        all_steps = 0
        for i_episode in range(total_episode):
            # Initialize the environment and state
            state =env.reset()
            total_reward = 0
            for t in range(0,1000):
                # Select and perform an action
                action = self.take_action(state)
                new_state, reward, done, _ = env.step(action)
                reward = torch.tensor([reward], device=self.device)
                new_state = torch.tensor([state], device=self.device)
                total_reward+=reward
                
                # Store the transition in memory
                self.add_to_memory(state, action, new_state, reward,done)
                # Move to the next state
                state = new_state
                # Perform one step of the optimization
                self.online_training(state, action, reward, new_state, done)
                if done:
                    break
            all_rewards += total_reward
            all_steps += t + 1
            self.reward_list.append(total_reward)
        print('Average reward is {}, average step is {}'.
            format(all_rewards / total_episode, all_steps / total_episode))

        
    def eval(self, env, total_episode):
        """Evaluate the approximator.

        Args:
            env (gym.Environment): the environment that provides gym.Environment interface.
            total_episode (int): the number of episodes to evaluate.

        Returns:
            reward_list (list[float]): the list of rewards for every episode.
        """
        reward_list = []
        # Training has ended; thus agent does not need to explore.
        # However, you can leave it unchanged and it may not make much difference here.
        self.epsilon = 0.0
        # begin answer
        
        # end answer
        print('Average reward per episode is {}'.format(sum(reward_list) / total_episode))
        # change epsilon back for training
        self.epsilon = self.min_epsilon
        return reward_list
import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from IPython.display import display, clear_output
import time
import random
from approximator_torch import Approximator
env = gym.make('CartPole-v1')
gamma = 0.95 # discounting rate
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
print(state_size,action_size)
# tune the learning_rate and total_episode yourself
# remember how learning_rate can influence the training for NNs?
learning_rate = 0.01
total_episode = 100
# begin answer
# end answer
model = Approximator(state_size, action_size,
                     learning_rate=learning_rate, memory_pool_size=10000, batch_size=20)

model.train(env, total_episode)
# plot to see the training rewards of each episode
plt.plot(np.arange(len(model.reward_list)), np.array(model.reward_list))
plt.show()