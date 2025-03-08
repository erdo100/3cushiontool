import pickle
import time
import numpy as np
from scipy.interpolate import interp1d
from tkinter import filedialog
from slider_definitions import create_ballball_sliders, create_physics_sliders
  
def read_shotfile():

    # pick file woth UI
    file_path = r"E:\PYTHON_PROJECTS\POOLTOOL\3cushiontool\Scripts\20221225_2_Match_Ersin_Cemal.pkl"
    file_path = filedialog.askopenfilename()

    # Load shots from the pickle file
    with open(file_path, "rb") as f:
        shots_actual = pickle.load(f)

    # Swap x and y axes
    for shot_actual in shots_actual:
        for ball in shot_actual["balls"].values():
            ball["x"], ball["y"] = np.array(ball["y"]), np.array(ball["x"])

    return shots_actual

# Function to save parameters
def save_parameters(ballball_hit_params, physics_params):
    params = {
        'ballball_hit_params': ballball_hit_params,
        'physics_params': physics_params
    }
    filename = "parameters_" + time.strftime("%Y%m%d_%H%M%S") + ".pkl"
    file_path = filedialog.asksaveasfilename(initialfile=filename ,defaultextension='.pkl', filetypes=[('Pickle files', '*.pkl')])
    if file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
        print(f"Parameters saved to {file_path}")

# Function to save parameters
def save_system(update_plot):
    system = update_plot()
    
    filename = "system_" + time.strftime("%Y%m%d_%H%M%S") + ".msgpack"
    
    file_path = filedialog.asksaveasfilename(initialfile=filename ,defaultextension='.pkl', filetypes=[('Pickle files', '*.pkl')])
    if file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(system, f)
        print(f"System saved to {file_path}")
        system.save("Ball_dislocation_bug.msgpack")

def load_parameters(slider_frame, update_plot, ballball_a_slider, ballball_b_slider, ballball_c_slider, physics_u_slide_slider, physics_u_roll_slider, physics_u_sp_prop_slider, physics_e_ballball_slider, physics_e_cushion_slider, physics_f_cushion_slider, physics_cushion_height):
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if file_path:
        with open(file_path, 'rb') as f:
            params = pickle.load(f)

        ballball_hit_params = params['ballball_hit_params']
        physics_params = params['physics_params']

        ballball_a_slider.set(ballball_hit_params['a'])
        ballball_b_slider.set(ballball_hit_params['b'])
        ballball_c_slider.set(ballball_hit_params['c'])
        physics_u_slide_slider.set(physics_params['u_slide'])
        physics_u_roll_slider.set(physics_params['u_roll'])
        physics_u_sp_prop_slider.set(physics_params['u_sp_prop'])
        physics_e_ballball_slider.set(physics_params['e_ballball'])
        physics_e_cushion_slider.set(physics_params['e_cushion'])
        physics_f_cushion_slider.set(physics_params['f_cushion'])
        physics_cushion_height.set(physics_params['h_cushion'])

        update_plot()   

def loss_func(actual_x, actual_y, simulated_x, simulated_y):
    distances = np.sqrt((actual_x - simulated_x) ** 2 + (actual_y - simulated_y) ** 2)
    rms = np.sqrt(np.mean(distances ** 2))
    return rms

def interpolate_simulated_to_actual(simulated, tsim, actual_times):
    interp_func = interp1d(
        tsim,
        simulated,
        kind="linear",
        bounds_error=False,
        fill_value=(simulated[-1]),
    )
    interpolated = interp_func(actual_times)
    return interpolated

# calculate absolute velocity
def calculate_velocity(x, y, t):
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    v = np.sqrt(dx ** 2 + dy ** 2) / dt
    # add a zero to the beginning of the array to make it the same length as x and y
    v = np.insert(v, 0, 0)
    return v