'''
Main file for the reinforcement learning trading algorithm.
'''
import tkinter as tk
from trading_environment import TradingEnvironment
from trading_agent import TradingAgent
from gui import GUI
def main():
    # Initialize the trading environment
    environment = TradingEnvironment(historical_data='historical_data.csv')
    # Initialize the trading agent with the state and action sizes
    state_size = environment.get_state_size()
    action_size = environment.get_action_size()
    agent = TradingAgent(environment, state_size, action_size)
    # Initialize the GUI
    app = GUI(agent)
    app.mainloop()
if __name__ == "__main__":
    main()