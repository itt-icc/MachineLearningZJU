from collections import deque
import numpy as np
import random
import tensorflow as tf

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
        # begin answer
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
        # end answer
        pass
    
    def take_action(self, state):
        """Determine the action for state according to Q-value and epsilon-greedy strategy.
        
        Args:
            state (int): the current state.

        Returns:
            action (int): the action to take.
        """
        action = 0
        # begin answer
        # end answer
        return action
    
    def online_training(self, state, action, reward, new_state, done):
        """Train the approximator with only one sample.

        Args:
            state (int): the current state.
            action (int): the action to take.
            reward (int): the reward corresponding to state and action.
            new_state (int): the new state after taking action.
            done (bool): whether the decision process ends in this state.
        """
        # begin answer
        # end answer
        pass
    
    def experience_replay(self):
        """Use experience replay to train the approximator.
        """
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
        pass

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
        all_rewards = 0
        # begin answer
        # end answer
        print('Average reward per episode is {}'.format(all_rewards / total_episode))
        # change epsilon back for training
        self.epsilon = self.min_epsilon
        return reward_list