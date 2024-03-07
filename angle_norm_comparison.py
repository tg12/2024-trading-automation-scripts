import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Initialize variables
data_vec = np.array([1, 4])
N_grid = 201
param_grid = np.linspace(0, 2*np.pi, N_grid)

# Define sine-cosine matrix function
def sin_cos_mat(x):
    return np.column_stack((np.sin(x), np.cos(x)))

# Define norms
norms = {
    'L1': lambda x: np.abs(x),
    'L2': lambda x: x**2,
}

# Prepare for data collection
loss_results = []

# Main analysis loop
for data_val in data_vec:
    d = np.abs(data_val - param_grid)
    angle_dist = np.where(d > np.pi, 2*np.pi - d, d)
    for norm_name, norm_fun in norms.items():
        sin_cos_loss = np.sum(norm_fun(np.abs(sin_cos_mat(param_grid) - sin_cos_mat(np.full(N_grid, data_val)))), axis=1)
        for loss_name, loss_values in [('angle', norm_fun(angle_dist)), ('sin_cos', sin_cos_loss)]:
            for grid_val, loss_val in zip(param_grid, loss_values):
                loss_results.append([data_val, norm_name, grid_val, loss_name, loss_val])

# Convert results to DataFrame
loss_df = pd.DataFrame(loss_results, columns=['data_val', 'norm_name', 'param_grid', 'loss_name', 'loss_value'])

# Plotting
plt.figure(figsize=(10, 6))
g = sns.FacetGrid(loss_df, row='norm_name', col='data_val', hue='loss_name', margin_titles=True, sharey=False)
g.map(sns.scatterplot, 'param_grid', 'loss_value')
g.add_legend()

# Save the plot
plt.savefig('figure-loss-python.png')
plt.show()
