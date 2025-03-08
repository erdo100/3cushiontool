import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pooltool as pt
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pooltool.ruleset.three_cushion import is_point
from billiardenv import BilliardEnv
from helper_funcs import get_ball_ids, get_ball_positions, read_shotfile, interpolate_simulated_to_actual, loss_func, save_parameters, load_parameters, save_system, calculate_velocity, initial_shot_direction
from slider_definitions import create_sliders


def plot_settings(ax):
    ax.cla()  # Clear the current axis
    ax.set_xlim(0.0, 1.42)
    ax.set_ylim(0.0, 2.84)
    ax.set_aspect("equal", adjustable="box")  # Set the aspect ratio to 1:2
    ax.set_facecolor("lightgray")  # Set the background color to light gray
    grid_size = 2.84 / 8
    ax.set_xticks(np.arange(0, 1.42 + grid_size, grid_size))
    ax.set_yticks(np.arange(0, 2.84 + grid_size, grid_size))
    ax.grid(True)


def plot_initial_positions(ax, ball_xy_ini):
    for ball_col, ball_xy in ball_xy_ini.items():
        circle = plt.Circle(ball_xy, 0.0615 / 2, color=ball_col, fill=True)
        ax.add_patch(circle)

def plot_current_shot(ax, colind, actual_x, actual_y, simulated_x, simulated_y):
    color_mapping = {1: "white", 2: "yellow", 3: "red"}
    colname = color_mapping[colind]
    circle = plt.Circle((simulated_x[0], simulated_y[0]), 0.0615 / 2, color=colname, fill=True)
    ax.add_patch(circle)
    ax.plot(actual_x, actual_y, "--", color=colname, linewidth=1)
    ax.plot(simulated_x, simulated_y, "-", color=colname, linewidth=1)

def update_shot_id(event=None):
    global shot_id_changed
    shot_id_changed = True

    update_plot()

# Convert pixels to figure coordinates
def px_to_fig_x(px):
    return px / (fig.get_figwidth() * fig.dpi)

def px_to_fig_y(px):
    return px / (fig.get_figheight() * fig.dpi)

def update_plot(event=None):
    global cueball_id, a, b, phi, v, theta, a_ballball, b_ballball, c_ballball, u_slide, u_roll
    global u_sp_prop, e_ballball, e_cushion, f_cushion, h_cushion, shot_actual, ball1xy, ball2xy, ball3xy, cue_phi
    global shot_id_changed

    # Retrieve current shot and slider values
    shot_id = params['shot_id'] = sliders['shot_id'].get()
    shot_actual = shots_actual[shot_id]
    ball_xy_ini, ball_ids, ball_cols, cue_phi = get_ball_positions(shot_actual)

    a = params['a'] = sliders['shot_a'].get()
    b = params['b'] = sliders['shot_b'].get()

    if shot_id_changed:
        phi = params['phi'] = cue_phi
        sliders['shot_phi'].set(cue_phi)
        shot_id_changed = False
        print("update_plot: new shot", phi)
    else:
        phi = params['phi'] = sliders['shot_phi'].get()
        print("update_plot: current shot", phi)
        shot_id_changed = False


    v = params['v'] = sliders['shot_v'].get()
    theta = params['theta'] = sliders['shot_theta'].get()

    a_ballball = params['ballball_a'] = sliders['ballball_a'].get()
    b_ballball = params['ballball_b'] = sliders['ballball_b'].get()
    c_ballball = params['ballball_c'] = sliders['ballball_c'].get()

    u_slide = params['physics_u_slide'] = sliders['physics_u_slide'].get()
    u_roll = params['physics_u_roll'] = sliders['physics_u_roll'].get()
    u_sp_prop = params['physics_u_sp_prop'] = sliders['physics_u_sp_prop'].get()
    e_ballball = params['physics_e_ballball'] = sliders['physics_e_ballball'].get()
    e_cushion = params['physics_e_cushion'] = sliders['physics_e_cushion'].get()
    f_cushion = params['physics_f_cushion'] = sliders['physics_f_cushion'].get()
    h_cushion = params['physics_h_cushion'] = sliders['physics_h_cushion'].get()

    
    # Create billiard environment and simulate shot
    shot_env = BilliardEnv(u_slide, u_roll, u_sp_prop, e_ballball, e_cushion, f_cushion, h_cushion)
    shot_env.prepare_new_shot(ball_cols, ball_xy_ini, a, b, phi, v, theta)
    system = shot_env.simulate_shot(a_ballball, b_ballball, c_ballball)
    tsim, white_rvw, yellow_rvw, red_rvw = shot_env.get_ball_routes()

    # Clear entire figure and create a new grid layout.
    fig.clf()

    # Left side (main plot)
    left_margin_px = 5
    bottom_margin_px = 5
    right_margin_px = 5
    top_margin_px = 5
    gap_px = 50  # Gap between the three right-side diagrams
    
    # Convert margins to figure-relative coordinates
    left_margin = px_to_fig_x(left_margin_px)
    bottom_margin = px_to_fig_y(bottom_margin_px)
    right_margin = px_to_fig_x(right_margin_px)
    top_margin = px_to_fig_y(top_margin_px)
    gap = px_to_fig_y(gap_px)

    # Compute the width of the first diagram (normalized)
    first_width = 1/3
    first_height = 0.95  # Normalized height

    main_ax = fig.add_axes([left_margin, bottom_margin, first_width, first_height])
    
    plot_settings(main_ax)
    main_ax.set_xticklabels([])  # Remove x-axis tick labels
    main_ax.set_yticklabels([])  # Remove y-axis tick labels
    
    # Right side: Create three vertically arranged subplots with double width.
    # Compute available height for the 3 stacked diagrams on the right
    available_height = 1 - top_margin - bottom_margin - 2 * gap  # Space after top & bottom margins + gaps
    right_height = available_height / 3  # Equal height for 3 diagrams
    right_width = 0.6  # Given width

    # X position for right-side diagrams
    right_x = left_margin + first_width + px_to_fig_x(50)  # Offset by 50px

    # Create 3 right-side diagrams stacked vertically
    bottom_y = bottom_margin + 0 * (right_height + gap)
    ax_bot = fig.add_axes([right_x, bottom_y, right_width, right_height])
    bottom_y = bottom_margin + 1 * (right_height + gap)
    ax_mid = fig.add_axes([right_x, bottom_y, right_width, right_height])
    bottom_y = bottom_margin + 2 * (right_height + gap)
    ax_top = fig.add_axes([right_x, bottom_y, right_width, right_height])
    
    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.07, wspace=0.07, left=0.05)  # Adjust these values as needed

    total_loss = 0.0
    for ballx in [0, 1, 2]:
        # Select the appropriate rvw based on ball ID
        if ball_ids[ballx] == 1:
            rvw = white_rvw
        elif ball_ids[ballx] == 2:
            rvw = yellow_rvw
        elif ball_ids[ballx] == 3:
            rvw = red_rvw

        actual_times = shot_actual["balls"][ball_ids[ballx]]["t"]
        actual_x = shot_actual["balls"][ball_ids[ballx]]["x"]
        actual_y = shot_actual["balls"][ball_ids[ballx]]["y"]
        simulated_x = rvw[:,0,0]
        simulated_y = rvw[:,0,1]
        interp_x = interpolate_simulated_to_actual(tsim, simulated_x, actual_times)
        interp_y = interpolate_simulated_to_actual(tsim, simulated_y, actual_times)
        total_loss += loss_func(actual_x, actual_y, interp_x, interp_y)
        
        # Plot the shot on the main axis
        plot_current_shot(main_ax, ball_ids[ballx], actual_x, actual_y, simulated_x, simulated_y)

        actual_v = calculate_velocity(actual_x, actual_y, actual_times)
        simulated_v = calculate_velocity(simulated_x, simulated_y, tsim)

        if ball_ids[ballx] == 1:
            ax_top.plot(actual_times, actual_v, label="actual", linestyle='--', linewidth=1)
            ax_top.plot(tsim, simulated_v, label="simulated", linestyle='-', linewidth=1)#, marker='o')
            ax_top.set_title("White Ball absolute velocity in m/s")
        elif ball_ids[ballx] == 2:
            ax_mid.plot(actual_times, actual_v, label="actual", linestyle='--', linewidth=1)
            ax_mid.plot(tsim, simulated_v, label="simulated", linestyle='-', linewidth=1)#, marker='o')
            ax_mid.set_title("Yellow Ball absolute velocity in m/s")

        elif ball_ids[ballx] == 3:
            ax_bot.plot(actual_times, actual_v, label="actual", linestyle='--', linewidth=1)
            ax_bot.plot(tsim, simulated_v, label="simulated", linestyle='-', linewidth=1)
            ax_bot.set_title("Red Ball absolute velocity in m/s")

    main_ax.set_title(f"ShotID: {shot_actual['shotID']}, Loss: {total_loss:.2f}\n"
                      f"a: {round(a, 2)}  b: {round(b, 2)}\n"
                      f"phi: {round(phi, 2)}  v: {round(v, 2)}  theta: {round(theta, 2)}")

    for ax in [ax_top, ax_mid, ax_bot]:
        ax.legend(loc="best")
        ax.grid(True)

    canvas.draw()  # Update the figure display
    return system

def on_closing():
    root.destroy()
    sys.exit()

def show_system():
    system = update_plot()
    pt.show(system)


shots_actual = read_shotfile()


# Shot Parameter
params = {
    'shot_id': 0,

    # Shot parameters
    'shot_a': 0.0,
    'shot_b': 0.0,
    'shot_phi': -81.0,
    'shot_v': 3.0,
    'shot_theta': 0.0,

    # Alciatori Ball-Ball hit model parameters
    # Friction fit curve u_b = a + b * exp(-c * v_rel) used in David Alciatore's TP A-14
    'ballball_a': 0.009951,
    'ballball_b': 0.108,
    'ballball_c': 1.088,

    # Physics parameters
    'physics_u_slide': 0.2,
    'physics_u_roll': 0.005,
    'physics_u_sp_prop': 10 * 2 / 5 / 9,
    'physics_e_ballball': 0.95,
    'physics_e_cushion': 0.9,
    'physics_f_cushion': 0.15,
    'physics_u_ballball': 0.05, # not relevant for Alciatori Ball-Ball hit model
    'physics_h_cushion': 0.037
}

# Select the current shot
shot_actual = shots_actual[0]
shot_id_changed = True

# GUI setup
root = Tk()
root.title("3-Cushion Shot Optimizer")
root.protocol("WM_DELETE_WINDOW", on_closing)
screen_width = root.winfo_screenwidth()  # Screen width in pixels
screen_height = root.winfo_screenheight()  # Screen height in pixels

res_dpi = 100
# Figure size
fig_width = int(screen_width /res_dpi * 0.75)
fig_height = int(screen_height/res_dpi * 0.8)
fig = plt.figure(figsize=(fig_width, fig_height), dpi=res_dpi)



# Create a frame for the plot
plot_frame = Frame(root)
plot_frame.pack(side="left", fill="both", expand=True)

# Create a canvas for the plot
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

# Add the toolbar for zooming and panning
toolbar = NavigationToolbar2Tk(canvas, plot_frame)
toolbar.update()

# Create a frame for the sliders
slider_frame = Frame(root)
slider_frame.pack(side="right", fill="both", expand=False)

# Initialize sliders dictionary
global sliders
sliders = {}

param_sliders = create_sliders(slider_frame, params, shots_actual, update_shot_id, update_plot)
sliders.update(param_sliders)

# Add a button to show the system
show_button = Button(slider_frame, text="Show System", command=show_system)
show_button.pack()

# Add a button to save the parameters
save_button = Button(slider_frame, text="Save Parameters", command=lambda: save_parameters(params))
save_button.pack()

# Add a button to load the parameters
load_button = Button(slider_frame, text="Load Parameters", command=lambda: load_parameters(slider_frame, update_plot, **sliders))
load_button.pack()

# Add a button to save the system
save_system_button = Button(slider_frame, text="Save System", command=lambda: save_system(update_plot))
save_system_button.pack()

root.mainloop()