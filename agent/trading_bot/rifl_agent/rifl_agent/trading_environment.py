'''
Trading environment class for the reinforcement learning trading algorithm.
This environment simulates trading Dow Jones futures on the CME. It manages the state of the portfolio, executes trades, and calculates rewards.
'''
import numpy as np
import pandas as pd
import datetime
class TradingEnvironment:
    def __init__(self, initial_balance=100000, aggressive_reward=True, historical_data=None):
        # Initialize environment state
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.aggressive_reward = aggressive_reward
        self.current_step = 0
        self.done = False
        self.position = 0  # Current position: 1 for long, -1 for short, 0 for flat
        self.inventory = []  # Stores the prices at which futures were bought or sold short
        self.prices = self._load_prices(historical_data)  # Load historical Dow Jones futures prices
        self.n_step = len(self.prices) - 1
        self.action_space = 4  # Number of possible actions: hold, long, short, exit
    def _load_prices(self, historical_data):
        # Load historical Dow Jones futures prices
        if historical_data is not None:
            prices = pd.read_csv(historical_data)
            return prices['Close'].values
        else:
            # This method should be replaced with a method to get real-time prices for live trading
            prices = np.random.rand(1000) * 10000  # Dummy prices for the sake of example
            return prices
    def reset(self):
        # Reset the environment state
        self.current_balance = self.initial_balance
        self.current_step = 0
        self.done = False
        self.position = 0
        self.inventory = []
        return self._get_observation()
    def step(self, action):
        # Apply the action to the environment and return the new state, reward, and done status
        # Action: 0 = hold, 1 = long, 2 = short, 3 = exit
        self.current_step += 1
        reward = 0
        current_price = self.prices[self.current_step]
        if action == 1 and self.position == 0:  # Go long
            self.position = 1
            self.inventory.append(current_price)
            reward = -1  # Penalize slightly to prevent excessive trading
        elif action == 2 and self.position == 0:  # Go short
            self.position = -1
            self.inventory.append(current_price)
            reward = -1  # Penalize slightly to prevent excessive trading
        elif action == 3 and self.position != 0:  # Exit
            if self.position == 1:  # Close long position
                bought_price = self.inventory.pop(0)
                reward = max(0, current_price - bought_price)  # Reward is the profit
                self.current_balance += current_price - bought_price
            elif self.position == -1:  # Close short position
                sold_price = self.inventory.pop(0)
                reward = max(0, sold_price - current_price)  # Reward is the profit
                self.current_balance += sold_price - current_price
            self.position = 0
        # Calculate additional reward for aggressive behavior if enabled
        if self.aggressive_reward:
            reward *= 2 if reward > 0 else 0.5
        # Check if the episode is done
        self.done = self.current_step >= self.n_step
        # Penalize the agent if it goes bankrupt
        if self.current_balance <= 0:
            self.done = True
            reward = -self.initial_balance
        return self._get_observation(), reward, self.done
    def _get_observation(self):
        # Return the current state as an observation
        # For simplicity, the observation is just the current price and position
        observation = (self.prices[self.current_step], self.position)
        return observation
    def render(self):
        # Render the environment to the screen (optional for GUI)
        # This could be a graph of the portfolio value over time or the current position
        pass