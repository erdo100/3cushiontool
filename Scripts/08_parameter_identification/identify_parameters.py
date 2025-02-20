import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from threecushion_shot import BilliardEnv

# Load shots from shots.pkl
with open('shots.pkl', 'rb') as f:
    shots = pickle.load(f)


# make data quality check of digitized data
# determine cue ball for current shot
# determine ball 2 for current shot
# Calculation of phi

# make data quality check of digitized data



# Swap x and y axes and scale by 1000
for shot in shots:
    for ball in shot['balls'].values():
        ball['x'], ball['y'] = np.array(ball['y']) / 1000, np.array(ball['x']) / 1000

# Define physics parameters and their limits
physics_params = {
    'u_slide': 0.15,
    'u_roll': 0.005,
    'u_sp_prop': 10 * 2 / 5 / 9,
    'u_ballball': 0.05,
    'e_ballball': 0.95,
    'e_cushion': 0.9,
    'f_cushion': 0.15
}
physics_limits = {
    'u_slide': (0.01, 0.2),
    'u_roll': (0.001, 0.02),
    'u_sp_prop': (0.1, 0.9),
    'u_ballball': (0.01, 0.2),
    'e_ballball': (0.8, 0.98),
    'e_cushion': (0.5, 0.95),
    'f_cushion': (0.01, 0.2)
}

# Create billiard environment
shot_env = BilliardEnv(**physics_params)

def initial_shot_direction(ball1_trajectory):
    dx = ball1_trajectory['x'][1] - ball1_trajectory['x'][0]
    dy = ball1_trajectory['y'][1] - ball1_trajectory['y'][0]
    phi = np.arctan2(dy, dx)
    print(dx, dy, phi)
    
    return np.degrees(phi)

def initial_cut_angle(ball1_trajectory, ball2_trajectory):
    dx = ball2_trajectory['x'][0] - ball1_trajectory['x'][0]
    dy = ball2_trajectory['y'][0] - ball1_trajectory['y'][0]
    phi = np.arctan2(dy, dx)
    print(dx, dy, phi)
    
    return np.degrees(phi)

def rms_difference(simulated, actual):
    return np.sqrt(np.mean((simulated - actual) ** 2))

def interpolate_simulated_to_actual(simulated, tsim, actual_times):
    interp_func_x = interp1d(tsim, simulated[:,0], kind='linear', fill_value="extrapolate")
    interp_func_y = interp1d(tsim, simulated[:,1], kind='linear', fill_value="extrapolate")
    interpolated_x = interp_func_x(actual_times)
    interpolated_y = interp_func_y(actual_times)
    return interpolated_x, interpolated_y

def plot_shot(actual_x, actual_y, simulated_x, simulated_y, ball_num):
    plt.figure()
    plt.plot(actual_x, actual_y, 'ro-', label='Actual')
    plt.plot(simulated_x, simulated_y, 'bo-', label='Simulated')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title(f'Ball {ball_num} Trajectory')
    plt.legend()
    plt.show()

def optimize_shot_parameters(shot, initial_params, ball1xy, ball2xy, ball3xy):
    def objective(params):
        a, b, cut, v, theta = params
        shot_env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, cut, v, theta)
        point, result, tsim = shot_env.simulate_shot()
        shot_env.plot_shot()

        rms_total = 0
        for ball_num in shot['balls']:
            actual_times = shot['balls'][ball_num]['t']
            actual_x = shot['balls'][ball_num]['x']
            actual_y = shot['balls'][ball_num]['y']
            simulated_x, simulated_y = interpolate_simulated_to_actual(result[ball_num], tsim, actual_times)
            rms_total += rms_difference(simulated_x, actual_x) + rms_difference(simulated_y, actual_y)
            print(rms_total, a, b, cut, v, theta)
            plot_shot(actual_x, actual_y, simulated_x, simulated_y, ball_num)  # Plot the trajectories
        return rms_total

    bounds = [(-0.5, 0.5),  # a
              (-0.5, 0.5),  # b
              (0, 359),     # cut
              (1, 6),       # v
              (0, 0)]      # theta
    result = minimize(objective, initial_params, bounds=bounds)
    return result.x, result.fun

def optimize_physics_parameters():
    def objective(params):
        for i, key in enumerate(physics_params.keys()):
            physics_params[key] = params[i]
                
        # Create billiard environment
        shot_env = BilliardEnv(**physics_params)
        
        total_rms = 0
        for shot in shots:
            ball1xy = (shot['balls'][1]['x'][0], shot['balls'][1]['y'][0])
            ball2xy = (shot['balls'][2]['x'][0], shot['balls'][2]['y'][0])
            ball3xy = (shot['balls'][3]['x'][0], shot['balls'][3]['y'][0])
            phi = initial_shot_direction(shot['balls'][1])
            print(phi)
            initial_params = [0.0, 0.0, phi, 3.0, 0]
            _, rms = optimize_shot_parameters(shot, initial_params, ball1xy, ball2xy, ball3xy)
            total_rms += rms
        return total_rms

    bounds = [physics_limits[key] for key in physics_params.keys()]
    initial_params = list(physics_params.values())
    result = minimize(objective, initial_params, bounds=bounds)
    return result.x, result.fun

# Optimize physics parameters
optimized_physics_params, total_rms = optimize_physics_parameters()
print(f"Optimized physics parameters: {optimized_physics_params}")
print(f"Total RMS after optimizing physics parameters: {total_rms}")
