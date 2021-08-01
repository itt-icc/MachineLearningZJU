import numpy as np
import random

class QTable:
    '''QTable for Q-Learning in reinforcement learning.
    
    Note that this class supports for solving problems that provide
    gym.Environment interface.
    '''

    def __init__(self,
                 state_size,
                 action_size,
                 alpha=0.8,
                 gamma=0.95,
                 init_epsilon=0.0,
                 epsilon_decay=0.995,
                 min_epsilon=0.0,
                 ):
        '''Initialize the approximator.

        Args:
            state_size (int): the number of states for this environment. 
            action_size (int): the number of actions for this environment.
            alpha (float): the learning rate for updating qtable.
            gamma (float): the gamma factor for reward decay.
            init_epsilon (float): the initial epsilon probability for exploration.
            epsilon_decay (float): the decay factor each step for epsilon.
            min_epsilon (float): the minimum epsilon in training.
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.qtable = np.zeros((self.state_size, self.action_size))
    
    def bellman_equation_update(self, state, action, reward, new_state):
        """Update the qtable according to the bellman equation.

        Args:
            state (int): the current state.
            action (int): the action to take.
            reward (int): the reward corresponding to state and action.
            new_state(int): the next state after taking action.
        """
        # begin answer
        self.qtable[state,action]+=self.alpha*(reward+self.gamma*np.max(self.qtable[new_state,:])-self.qtable[state, action])
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
        if random.random()<self.epsilon:
            action=random.randint(0,self.action_size-1)
        else:
            maxr=np.max(self.qtable[state,:])
            select=[i for i,r in enumerate(self.qtable[state,:]) if r == maxr ]
            action=random.choice(select)
        
        if self.epsilon>self.min_epsilon:
            self.epsilon*=self.epsilon_decay
        # end answer
        return action
    
    def set_epsilon(self, epsilon):
        """Set self.epsilon with epsilon"""
        self.epsilon = epsilon

    def train(self, env, total_episode, max_steps=100):
        """Train the QTable.
        Args:
            env (gym.Environment): the environment that provides gym.Environment interface.
            total_episode (int): the number of episodes to train.
            max_steps (int): max step to take for each episode.
        """
        # save the rewards for each training episode in self.reward_list.
        self.reward_list = []
        all_rewards = 0
        all_steps = 0
        
        for episode in range(total_episode):
            total_reward = 0
            state = env.reset()
            # print(f"episode={episode}")
            for step in range(max_steps):
            # begin answer
                action=self.take_action(state)
                new_state, reward, done, info = env.step(action)#observe
                # print(f"step = {step},reward = {reward}")
                self.bellman_equation_update(state, action, reward, new_state)#update
                total_reward+=reward
                state=new_state
                if done:
                    break
            # end answer
            all_rewards += total_reward
            all_steps += step + 1
            self.reward_list.append(total_reward)
        print('Average reward is {}, average step is {}'.
            format(all_rewards / total_episode, all_steps / total_episode))

    def eval(self, env, total_episode, max_steps=100):
        """Evaluate the QTable.

        Args:
            env (gym.Environment): the environment that provides gym.Environment interface.
            total_episode (int): the number of episodes to evaluate.
        """
        # Training has ended; thus agent does not need to explore.
        # However, you can leave it unchanged and it may not make much difference here.
        self.epsilon = 0.0
        all_rewards = 0
        all_steps = 0
        
        for episode in range(total_episode):
            total_reward = 0
            # reset the environment
            state = env.reset()
            for step in range(max_steps):
            # begin answer
                action=self.take_action(state)
                new_state, reward, done, info = env.step(action)
                state = new_state
                total_reward+=reward
                if done:break
            # end answer
            all_rewards += total_reward
            all_steps += step + 1
        
        print('Average reward is {}, average step is {}'.
            format(all_rewards / total_episode, all_steps / total_episode))
        # change epsilon back for training
        self.epsilon = self.min_epsilon
        