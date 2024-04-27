'''
Trading agent class for the reinforcement learning trading algorithm.
This agent interacts with the trading environment, decides actions to take (long, short, hold, exit),
and learns from the outcomes of those actions using a Deep Q-Network (DQN).
'''
import numpy as np
import random
from dqn import DQN
from replay_buffer import ReplayBuffer
class TradingAgent:
    def __init__(self, environment):
        self.environment = environment
        self.model = DQN()
        self.replay_buffer = ReplayBuffer()
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount rate for future rewards
    def act(self, state):
        # Decide an action based on the current state using an epsilon-greedy strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.environment.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    def train(self, batch_size):
        # Train the model using a batch of experience from the replay buffer
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        target_batch = []
        for i in range(0, batch_size):
            target = rewards[i]
            if not dones[i]:
                target = (rewards[i] + self.gamma * np.amax(self.model.predict(next_states[i])[0]))
            target_f = self.model.predict(states[i])
            target_f[0][actions[i]] = target
            target_batch.append(target_f)
        self.model.update(states, np.array(target_batch), self.learning_rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def load(self, name):
        self.model.load(name)
    def save(self, name):
        self.model.save(name)
# Additional methods for the DQN model, such as load and save, are assumed to be implemented in the dqn.py file.