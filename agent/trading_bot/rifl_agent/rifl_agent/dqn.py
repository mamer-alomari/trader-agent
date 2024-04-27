'''
Deep Q-Network (DQN) implementation for the reinforcement learning trading algorithm.
This class defines the neural network architecture, prediction, and update methods.
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    def predict(self, state):
        # Predict action values (Q-values) for the given state
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state)
        return q_values
    def update(self, states, targets, learning_rate):
        # Update the model using the given batch of states and targets
        self.model.fit(states, targets, epochs=1, verbose=0)
    def load(self, name):
        # Load model weights
        self.model.load_weights(name)
    def save(self, name):
        # Save model weights
        self.model.save_weights(name)