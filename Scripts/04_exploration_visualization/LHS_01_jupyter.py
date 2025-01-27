import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display # This is needed for Jupyter

# Load data (with dummy data for testing)
try:
    samples_df = pd.read_parquet("mein_dataframe.parquet")
except FileNotFoundError:
    print("Error: mein_dataframe.parquet not found. Creating dummy data.")
    samples_df = pd.DataFrame({
        'point': [1, 1, 1, 1, 1, 1],
        'a': [1, 2, 3, 1, 2, 3],
        'b': [4, 5, 6, 7, 8, 9],
        'cut': [10, 11, 12, 13, 14, 15]
    })

def plot_3d(xi_name, cut_value):
    # ... (rest of your plot_3d function remains the same)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ... (plotting logic)
    return fig

# Widgets erstellen
xi_dropdown = widgets.Dropdown(options=['cut', 'a', 'b'], description='Schnittachse:')
cut_slider = widgets.FloatSlider(min=samples_df['cut'].min(), max=samples_df['cut'].max(), step=0.01, value=samples_df['cut'].mean(), description='Schnittwert:')
a_slider = widgets.FloatSlider(min=samples_df['a'].min(), max=samples_df['a'].max(), step=0.01, value=samples_df['a'].mean(), description='Schnittwert:')
b_slider = widgets.FloatSlider(min=samples_df['b'].min(), max=samples_df['b'].max(), step=0.01, value=samples_df['b'].mean(), description='Schnittwert:')

def update_plot(change):
    fig = plot_3d(xi_dropdown.value, globals()[xi_dropdown.value + '_slider'].value)
    display(fig) # Display the figure within the notebook

# Initial plot
fig = plot_3d(xi_dropdown.value, cut_slider.value)
display(fig)

# Connect widgets and display
xi_dropdown.observe(update_plot, names='value')
cut_slider.observe(update_plot, names='value')
a_slider.observe(update_plot, names='value')
b_slider.observe(update_plot, names='value')

display(xi_dropdown)
display(cut_slider)
display(a_slider)
display(b_slider)