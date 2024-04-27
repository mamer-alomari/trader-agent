'''
This file handles the graphical user interface components and interactions for the Q-learning application.
'''
import tkinter as tk
from agent import Agent
from environment import Environment
from q_learning import QLearning
class GUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.environment = Environment(width=5, height=5, goal_position=(4, 4))
        self.agent = Agent(
            num_states=self.environment.num_states,
            num_actions=self.environment.num_actions
        )
        self.q_learning = QLearning(self.agent, self.environment, update_callback=self.update_environment_visual)
        self.update_environment_visual()
    def create_widgets(self):
        self.train_button = tk.Button(self)
        self.train_button["text"] = "Train Agent"
        self.train_button["command"] = self.train_agent
        self.train_button.pack(side="top")
        self.quit_button = tk.Button(self, text="QUIT", fg="red",
                                     command=self.master.destroy)
        self.quit_button.pack(side="bottom")
        self.canvas = tk.Canvas(self, width=250, height=250)
        self.canvas.pack(side="top")
    def update_environment_visual(self):
        # ... (rest of the update_environment_visual method code)
    def train_agent(self):
        self.q_learning.train()
    # ... (rest of the GUI class code)