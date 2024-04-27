'''
This file contains the implementation of the Q-learning algorithm.
The Q-learning class is responsible for training the agent within the environment.
It updates the Q-table based on the agent's experience and manages the learning process.
'''
import numpy as np
class QLearning:
    def __init__(self, agent, environment, update_callback=None):
        self.agent = agent
        self.environment = environment
        self.update_callback = update_callback
        self.q_table = np.zeros((self.environment.num_states, self.agent.num_actions))
    def train(self, episodes=1000, learning_rate=0.1, discount_factor=0.9):
        for episode in range(episodes):
            state = self.environment.reset()
            done = False
            while not done:
                action = self.agent.choose_action(state, self.q_table)
                next_state, reward, done = self.environment.step(action)
                self.update_q_table(state, action, reward, next_state, discount_factor, learning_rate)
                state = next_state
                if self.update_callback:
                    self.update_callback()
    def update_q_table(self, state, action, reward, next_state, discount_factor, learning_rate):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += learning_rate * td_error