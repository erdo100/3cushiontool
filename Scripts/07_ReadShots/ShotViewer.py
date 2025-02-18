import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
BALL_COLORS = {1: 'black', 2: 'yellow', 3: 'red'}
BALL_DIAMETER = 61.5 # in mm

class BilliardDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Billiard Shot Viewer")
        
        # File Selection Button
        self.load_button = tk.Button(root, text="Load Data File", command=self.load_data)
        self.load_button.pack()
        
        # Shot List Table
        self.shot_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=20, width=30)
        self.shot_listbox.pack(side=tk.LEFT, fill=tk.Y)
        self.shot_listbox.bind('<<ListboxSelect>>', self.update_plot)
        
        # Matplotlib Figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.all_shots = []
    
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"), ("JSON files", "*.json")])
        if not file_path:
            return
        
        # Load the file
        if file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                self.all_shots = pickle.load(f)
        elif file_path.endswith(".json"):
            with open(file_path, "r") as f:
                self.all_shots = json.load(f)
                # Convert lists back to NumPy arrays
                for shot in self.all_shots:
                    for ball in shot["balls"].values():
                        ball["t"] = np.array(ball["t"])
                        ball["x"] = np.array(ball["x"])
                        ball["y"] = np.array(ball["y"])
        
        # Populate the listbox with shots
        self.shot_listbox.delete(0, tk.END)
        for i, shot in enumerate(self.all_shots):
            self.shot_listbox.insert(tk.END, f"{shot['filename']}: {shot['shotID']}")
        
    def update_plot(self, event):
        selected_indices = self.shot_listbox.curselection()
        if not selected_indices:
            return
        
        self.ax.clear()
        
        for idx in selected_indices:
            shot = self.all_shots[idx]
            for ball_color, ball_data in shot["balls"].items():
                self.ax.plot(ball_data["x"], ball_data["y"], label=f"Ball {ball_color} (Shot {shot['shotID']})", color=BALL_COLORS[ball_color])
                
                # Plot initial position as a circle
                initial_x, initial_y = ball_data["x"][0], ball_data["y"][0]
                circle = plt.Circle((initial_x, initial_y), BALL_DIAMETER/2, color=BALL_COLORS[ball_color], fill=True)
                self.ax.add_patch(circle)
 
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    viewer = BilliardDataViewer(root)
    root.mainloop()
