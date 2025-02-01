from SALib.sample import sobol
import pandas as pd
import plotly.express as px
from threecushion_shot import BilliardEnv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.widgets import Slider, RadioButtons

runsims = False
calculate_density = True

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


def calculate_probability_density(
        df: pd.DataFrame,
        variables: list,  # List of variable column names (e.g., ['v1', 'v2'])
        result_col: str,  # Name of the result column (e.g., 'P')
        stddev_dict: dict,  # Maps variables to their standard deviations (e.g., {'v1': 0.5})
        dv: float,  # Fraction of the variable's range (e.g., 0.1 = 10% of the range)
        output_file: str = "density_result.parquet"
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

    # Prepare grid axes (no buffer)
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
    density = np.zeros_like(mesh[0], dtype=np.float64)

    # Calculate density contributions
    for _, row in df.iterrows():
        p = row[result_col]
        if p == 0:
            continue

        gaussians = []
        for i, var in enumerate(variables):
            mean = row[var]
            stddev = stddev_dict[var]
            g = norm.pdf(grid_axes[i], loc=mean, scale=stddev)
            gaussians.append(g.reshape([1 if j != i else -1 for j in range(len(variables))]))

        product = np.ones_like(gaussians[0])
        for g in gaussians:
            product = product * g
        density += p * product

    # Normalize the density
    dx = [grid[1] - grid[0] for grid in grid_axes]
    volume_element = np.prod(dx)
    normalized_density = density / (np.sum(density) * volume_element)

    # Save to Parquet
    grid_coords = {f"{var}_grid": grid.flatten() for var, grid in zip(variables, mesh)}
    grid_coords["density"] = normalized_density.flatten()
    pd.DataFrame(grid_coords).to_parquet(output_file)

    return grid_axes, normalized_density


if calculate_density == True:
    # Laden aus der Parquet-Datei
    shots_df = pd.read_parquet("2_17_shots.parquet")

    # Compute density
    grid_axes, density = calculate_probability_density(
        df=shots_df,
        variables=['side', 'vert', 'cut'],
        result_col='point',
        stddev_dict={'side': 0.02, 'vert': 0.02, 'cut': 3},
        dv=0.05
    )


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
    density_shape = [len(axis) for axis in grid_axes]
    density = df["density"].values.reshape(density_shape)

    return grid_axes, density


# Load density data from file:
grid_axes, density = load_density_from_parquet("density_result.parquet")

# Normalize the density
dx = [grid[1] - grid[0] for grid in grid_axes]
volume_element = np.prod(dx)
normalized_density = density / (np.sum(density) * volume_element)


# 2. Interactive 3D Visualization with Plotly
def plot_3d_density_interactive(grid_axes, density, variable_labels=None):
    """
    Visualize the 3D density using Plotly Volume.

    Args:
        grid_axes (list): List of 1D arrays (grid coordinates for each variable).
        density (np.ndarray): 3D density array.
        variable_labels (list): Optional labels for axes (e.g., ["v1", "v2", "v3"]).
    """
    # Create 3D coordinate grids
    X, Y, Z = np.meshgrid(*grid_axes, indexing='ij')

    # Flatten coordinates and density for Plotly
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    density_flat = density.flatten()

    # Default axis labels
    if variable_labels is None:
        variable_labels = [f"Variable {i + 1}" for i in range(len(grid_axes))]

    # Create Volume plot
    fig = go.Figure(data=go.Volume(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=density_flat,
        isomin=0.1 * density_flat.max(),
        isomax=density_flat.max(),
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


def standalone_slice_viewer(grid_axes, density, variables):
    """Interactive 2D slice viewer with dynamic slider labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Initial setup
    var_idx = 0
    slice_idx = len(grid_axes[var_idx]) // 2
    x, y = grid_axes[1], grid_axes[2]
    slice_data = density[slice_idx, :, :]
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
            data = density[idx, :, :]
            x, y = grid_axes[1], grid_axes[2]
            xl, yl = variables[1], variables[2]
        elif var_idx == 1:
            data = density[:, idx, :]
            x, y = grid_axes[0], grid_axes[2]
            xl, yl = variables[0], variables[2]
        else:
            data = density[:, :, idx]
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


variables=['side spin', 'vertical spin', 'cut angle']

standalone_slice_viewer(grid_axes, normalized_density, variables)

# Visualize interactively
plot_3d_density_interactive(
    grid_axes=grid_axes,
    density=normalized_density,
    variable_labels=variables  # Optional: ["Temperature", "Pressure", "Velocity"]
)