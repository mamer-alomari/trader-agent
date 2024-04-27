'''
This file serves as the entry point for the Q-learning application.
It initializes the GUI and starts the main application loop.
'''
import tkinter as tk
from gui import GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(master=root)
    app.mainloop()