from SALib.sample import sobol
import pandas as pd
import plotly.express as px
from threecushion_shot import BilliardEnv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator

# Define the problem: 4 variables, each with 3 levels
problem = {
    'num_vars': 3,
    'names': ['a', 'b', 'cut'],  # Names for the variables
    # Bounds for each variable
    'bounds': [[-0.5, 0.5], # a
               [-0.5, 0.5], # b
               #[2, 7],      # velocity
               [-89, 89]    # cut angle
               ]
}

runs = 2**12

# method = 'Sobol'
method = 'LHS'

if method == 'Sobol':
    # Generate the Sobol design using Sobol sampling
    samples = sobol.sample(problem, runs)

elif method == 'LHS':
    # Generate random uniform samples within the bounds for each variable
    samples = np.random.uniform(
        low=[-0.5, -0.5, -89],  # Lower bounds for each variable
        high=[0.5, 0.5, 89],     # Upper bounds for each variable
        size=(runs, problem['num_vars'])  # Number of samples and variables
    )


# Convert to a pandas DataFrame for easier inspection
samples_df = pd.DataFrame(samples, columns=problem['names'])

# Print the size (number of samples and number of variables)
print("Size of the samples array:", samples.shape)


env = BilliardEnv()

# start measure runtime
import time
start = time.time()

for i in range(runs):
    a = samples_df.loc[i, 'a']
    b = samples_df.loc[i, 'b']
    v = 3.5
    cut = samples_df.loc[i, 'cut']
    theta = 0

    ball1xy = (0.5275, 0.71)
    ball2xy = (0.71, 0.71)
    ball3xy = (0.71, 2.13)

    env.prepare_new_shot(ball1xy, ball2xy, ball3xy, a, b, v, cut, theta)

    point = env.simulate_shot()
    samples_df.at[i, 'point'] = point

    if (i+1) % 200000 == 0:
        print((time.time() - start)/3600,"h: ", i ," runs")
        filtered_df = samples_df[samples_df['point'] == 1]
        # Create an interactive 3D scatter plot
        fig = px.scatter_3d(filtered_df, x='a', y='cut', z='b', title="Total runs=" + str(i))
        fig.show()


# print runtime
print("Runtime of the program is", time.time() - start)
print("Runtime of the program is", time.time() - start)
# Speichern im Parquet-Format
samples_df.to_parquet("mein_dataframe.parquet")
print('Dataframe saved to parquet file')

# Laden aus der Parquet-Datei
samples_df = pd.read_parquet("mein_dataframe.parquet")
print(samples_df)

# Auswahl der Achse für die Interpolation (xi)
xi_name = 'cut'  # Hier wird 'cut' als xi verwendet. Ändere dies nach Bedarf ('a' oder 'b')
xi = samples_df[xi_name].values

# Filtern der Daten für point == 1
samples_df_1 = samples_df[samples_df['point'] == 1]

# Scatterplot (nur Punkte mit point == 1)
fig, ax = plt.subplots(figsize=(8, 6))

if not samples_df_1.empty: # Überprüfen, ob überhaupt Punkte mit point == 1 vorhanden sind.
    ax.scatter(samples_df_1['a'], samples_df_1['b'], c='red', marker='x', label='Point 1', alpha=0.5)

    # Interpolation mit cKDTree und LinearNDInterpolator (nur für point == 1)
    x_grid, y_grid = np.mgrid[samples_df_1['a'].min():samples_df_1['a'].max():100j, samples_df_1['b'].min():samples_df_1['b'].max():100j]
    points_1 = samples_df_1[['a', 'b']].values
    xi_values_1 = samples_df_1[xi_name].values

    if len(points_1) > 0:
        interp = LinearNDInterpolator(points_1, xi_values_1)
        zi = interp(x_grid, y_grid)
        if zi is not None: # Überprüfen, ob die Interpolation erfolgreich war
            contour = ax.contour(x_grid, y_grid, zi, levels=10, colors='green', linewidths=0.5)
            ax.clabel(contour, inline=True, fontsize=8)
            #oder
            #im = ax.imshow(zi, extent=[samples_df_1['a'].min(), samples_df_1['a'].max(), samples_df_1['b'].min(), samples_df_1['b'].max()], origin='lower', cmap='viridis', alpha=0.5, aspect="auto")
            #fig.colorbar(im, ax=ax, label=xi_name)


# Beschriftungen und Legende
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_title('Scatterplot und Interpolation von ' + xi_name + ' (nur Point 1)')
ax.legend()
plt.tight_layout()
plt.show()

# # Create an interactive 3D scatter plot
# fig = px.scatter_3d(filtered_df, x='a', y='cut', z='b', title="3D Scatter Plot of a, b, cut")
# fig.show()


# Optionally, save the design to a CSV file for further analysis
#samples_df.to_csv('lhs_design_4vars_3levels.csv', index=False)