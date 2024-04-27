'''
This file defines the agent that will learn from the environment using the Q-learning algorithm.
An agent in Q-learning has to maintain a Q-table that stores the Q-values for each state-action pair,
update this table based on the rewards received from the environment, and choose actions based on this table.
'''
import numpy as np
import random
class Agent:
    def __init__(self, num_states, num_actions, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon # exploration rate
        self.model = None # Q-table initialized to None
        self.replay_buffer = [] # replay buffer for experience replay
        self.gamma = 0.95 # discount factor
        self.learning_rate = 0.001 # learning rate
        # for model training
        self.model = self._build_model() # initialize Q-table model
    def choose_action(self, state, q_table):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore
        else:
            return np.argmax(q_table[state])  # Exploit
    # ... (rest of the Agent class code)
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
    def _build_model(self):
        # Build a neural network model for Q-learning        model = Sequential()

        model.add(Dense(24, input_dim=self.num_states, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def update(self, state, action, reward, next_state, done):
        # Update the Q-table based on the agent's experience
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error


    def get_state_index(self, state):
        return self.state_dict[state]

    def get_action_index(self, action):
        return self.action_dict[action]

    def get_q_value(self, state, action):
        return self.q_table[state][action]
    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error



    def load(self, name):
        self.model.load(name)
    def save(self, name):
        self.model.save(name)