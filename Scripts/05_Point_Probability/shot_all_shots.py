from SALib.sample import sobol
import pandas as pd
import plotly.express as px
from threecushion_shot import BilliardEnv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.widgets import Slider, RadioButtons


def calculate_probability_map(
        df: pd.DataFrame,
        variables: list,  # List of variable column names (e.g., ['v1', 'v2'])
        result_col: str,  # Name of the result column (e.g., 'P')
        stddev_dict: dict,  # Maps variables to their standard deviations (e.g., {'v1': 0.5})
        dv: float,  # Fraction of the variable's range (e.g., 0.1 = 10% of the range)
) -> tuple:
    """
    Calculate the probability density for each point in the n-dimensional space.
    Returns:
    tuple: (grid_axes, density_array)
    """
    # Validate inputs
    missing_vars = [var for var in variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing variables: {missing_vars}")
    if result_col not in df.columns:
        raise ValueError(f"Result column '{result_col}' not found.")
    missing_stddev = [var for var in variables if var not in stddev_dict]
    if missing_stddev:
        raise ValueError(f"Missing std deviations for: {missing_stddev}")


    # Prepare grid axes
    grid_axes = []
    for var in variables:
        min_val = df[var].min()
        max_val = df[var].max()
        range_var = max_val - min_val

        # Handle edge case where all values are identical
        if range_var == 0:
            grid = np.array([min_val])
        else:
            step = dv * range_var  # Step size as fraction of the range
            grid = np.arange(min_val, max_val + step, step)

        grid_axes.append(grid)

    # Create n-dimensional grid
    mesh = np.meshgrid(*grid_axes, indexing='ij')
    total_probability = np.zeros_like(mesh[0], dtype=np.float64)

    results = df[result_col].values

    # loop over all mesh grid points
    for idx in np.ndindex(mesh[0].shape):

        # Initialize the probability
        P = np.ones_like(results)
        # calculate the total probability for all shots in the results
        for vari, var in enumerate(variables):
            m = mesh[vari][idx]
            stddev = stddev_dict[var]
            p = norm.pdf(df[var], loc=m, scale=stddev)
            # normalize the density and multiply with the probability
            P = P*p # probability density of all variables, without points

        # Normalize the total probablity
        P = P / np.sum(P)

        total_probability[idx] = np.sum(P * results)

        print(grid_axes[0][idx[0]], grid_axes[1][idx[1]], grid_axes[2][idx[2]], total_probability[idx])


    # Save to Parquet
    grid_coords = {f"{var}_grid": grid.flatten() for var, grid in zip(variables, mesh)}
    grid_coords["total_probability"] = total_probability.flatten()

    # Save the grid coordinates and total probability to a Parquet file
    print(f"Saving grid coordinates and total probability ...")
    pd.DataFrame(grid_coords).to_parquet("total_probability.parquet")

    return grid_axes, total_probability


def load_density_from_parquet(parquet_path: str) -> tuple:
    """
    Load the saved Parquet file and reconstruct grid axes and density array.

    Args:
        parquet_path (str): Path to the saved Parquet file.

    Returns:
        tuple: (grid_axes, density_array)
            - grid_axes: List of 1D arrays for each variable's grid.
            - density_array: N-dimensional numpy array of densities.
    """
    # Read the Parquet file
    df = pd.read_parquet(parquet_path)

    # Extract variable names (assumes grid columns end with "_grid")
    grid_cols = [col for col in df.columns if col.endswith("_grid")]
    variables = [col.replace("_grid", "") for col in grid_cols]

    # Reconstruct grid axes from unique values in each grid column
    grid_axes = []
    for var_grid in grid_cols:
        # Extract unique values and sort them to reconstruct the original grid
        grid = np.sort(df[var_grid].unique())
        grid_axes.append(grid)

    # Reshape the density column into the original n-dimensional array
    shape = [len(axis) for axis in grid_axes]
    total_probability = df["total_probability"].values.reshape(shape)

    return grid_axes, total_probability


# 2. Interactive 3D Visualization with Plotly
def plot_3d_probability_interactive(grid_axes, total_probability, variable_labels=None):
    """
    Visualize the 3D density using Plotly Volume.

    Args:
        grid_axes (list): List of 1D arrays (grid coordinates for each variable).
        density (np.ndarray): 3D density array.
        variable_labels (list): Optional labels for axes (e.g., ["v1", "v2", "v3"]).
    """
    # Create 3D coordinate grids
    X, Y, Z = np.meshgrid(*grid_axes, indexing='ij')

    # Flatten coordinates and total_probability for Plotly
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    P_flat = total_probability.flatten()

    # Default axis labels
    if variable_labels is None:
        variable_labels = [f"Variable {i + 1}" for i in range(len(grid_axes))]

    # Create Volume plot
    fig = go.Figure(data=go.Volume(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=P_flat,
        isomin=0.1 * P_flat.max(),
        isomax=P_flat.max(),
        opacity=0.1,
        surface_count=20,
        colorscale='viridis',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=variable_labels[0],
            yaxis_title=variable_labels[1],
            zaxis_title=variable_labels[2],
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.show()


def standalone_slice_viewer(grid_axes, total_probability, variables):
    """Interactive 2D slice viewer with dynamic slider labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Initial setup
    var_idx = 0
    slice_idx = len(grid_axes[var_idx]) // 2
    x, y = grid_axes[1], grid_axes[2]
    slice_data = total_probability[slice_idx, :, :]
    contour = ax.contourf(x, y, slice_data.T, cmap='viridis', levels=20)
    cbar = fig.colorbar(contour, ax=ax)
    ax.set_xlabel(variables[1])
    ax.set_ylabel(variables[2])

    # Slider axis
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label=f'{variables[var_idx]}',  # Initialize with current variable
        valmin=grid_axes[var_idx].min(),
        valmax=grid_axes[var_idx].max(),
        valinit=grid_axes[var_idx][slice_idx],
        valstep=np.diff(grid_axes[var_idx])[0]
    )

    # Radio buttons
    ax_radio = plt.axes([0.8, 0.1, 0.15, 0.15])
    radio = RadioButtons(ax_radio, labels=variables, active=0)

    def update_slice(val):
        """Update the plot with new slice position."""
        nonlocal var_idx
        idx = np.abs(grid_axes[var_idx] - val).argmin()

        if var_idx == 0:
            data = total_probability[idx, :, :]
            x, y = grid_axes[1], grid_axes[2]
            xl, yl = variables[1], variables[2]
        elif var_idx == 1:
            data = total_probability[:, idx, :]
            x, y = grid_axes[0], grid_axes[2]
            xl, yl = variables[0], variables[2]
        else:
            data = total_probability[:, :, idx]
            x, y = grid_axes[0], grid_axes[1]
            xl, yl = variables[0], variables[1]

        ax.clear()
        ax.contourf(x, y, data.T, cmap='viridis', levels=20)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f'Slice at {variables[var_idx]} = {grid_axes[var_idx][idx]:.2f}')
        fig.canvas.draw_idle()

    def select_variable(label):
        """Update slider and label when variable changes."""
        nonlocal var_idx
        var_idx = variables.index(label)

        # Update slider properties
        slider.valmin = grid_axes[var_idx].min()
        slider.valmax = grid_axes[var_idx].max()
        slider.valstep = np.diff(grid_axes[var_idx])[0]
        slider.set_val(grid_axes[var_idx][len(grid_axes[var_idx]) // 2])

        # Update slider label to show current variable
        slider.label.set_text(variables[var_idx])  # Fix: Update label text
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        fig.canvas.draw_idle()

        # Update plot
        update_slice(slider.val)

    # Connect events
    slider.on_changed(update_slice)
    radio.on_clicked(select_variable)

    plt.show()


runsims = False
calculate_density = False

if runsims == True:
    # Define the problem: 4 variables, each with 3 levels
    problem = {
        'num_vars': 3,
        'names': ['side', 'vert', 'cut'],  # Names for the variables
        # Bounds for each variable
        'bounds': [[-0.5, 0.5], # a
                   [-0.5, 0.5], # b
                   #[2, 7],      # velocity
                   [-89, 89]    # cut angle
                   ]
    }


    resolution = 2**17
    runs = resolution*(2*3+2)
    method = 'Sobol'
    # method = 'UniformRandom'

    if method == 'Sobol':
        # Generate the Sobol design using Sobol sampling
        samples = sobol.sample(problem, resolution)

    elif method == 'UniformRandom':
        # Generate random uniform samples within the bounds for each variable
        samples = np.random.uniform(
            low=[-0.5, -0.5, -89],  # Lower bounds for each variable
            high=[0.5, 0.5, 89],     # Upper bounds for each variable
            size=(runs, problem['num_vars'])  # Number of samples and variables
        )


    # Convert to a pandas DataFrame for easier inspection
    shots_df = pd.DataFrame(samples, columns=problem['names'])

    # Print the size (number of samples and number of variables)
    print("Size of the samples array:", samples.shape)


    env = BilliardEnv()

    # start measure runtime
    import time
    start = time.time()

    for i in range(runs):
        a = shots_df.loc[i, 'side']
        b = shots_df.loc[i, 'vert']
        v = 3.5
        cut = shots_df.loc[i, 'cut']
        theta = 0

        ball1xy = (0.5275, 0.71)
        ball2xy = (0.71, 0.71)
        ball3xy = (0.71, 2.13)

        env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, v, cut, theta)

        point = env.simulate_shot()
        shots_df.at[i, 'point'] = point

        if (i+1) % 200000 == 0:
            print((time.time() - start)/3600,"h: ", i ," runs")
            filtered_df = shots_df[shots_df['point'] == 1]
            # Create an interactive 3D scatter plot
            fig = px.scatter_3d(filtered_df, x='side', y='cut', z='vert', title="Total runs=" + str(i))
            fig.show()

    # print runtime
    print("Runtime of the program is", time.time() - start)

    # Speichern im Parquet-Format
    shots_df.to_parquet("2_17_shots.parquet")
    print('Dataframe saved to parquet file')


if calculate_density == True:
    # Laden aus der Parquet-Datei
    shots_df = pd.read_parquet("2_17_shots.parquet")

    # Compute density
    grid_axes, total_probability = calculate_probability_map(
        df=shots_df,
        variables=['side', 'vert', 'cut'],
        result_col='point',
        stddev_dict={'side': 0.02, 'vert': 0.02, 'cut': 3},
        dv=0.05
    )
else:
    # Load density data from file:
    grid_axes, total_probability = load_density_from_parquet("total_probability.parquet")


variables=['side spin', 'vertical spin', 'cut angle']

# Visualize interactively
plot_3d_probability_interactive(
    grid_axes=grid_axes,
    total_probability=total_probability,
    variable_labels=variables  # Optional: ["Temperature", "Pressure", "Velocity"]
)

standalone_slice_viewer(grid_axes, total_probability, variables)
