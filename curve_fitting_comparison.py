
# Disclaimer:
# This script is provided for educational and informational purposes only. It demonstrates the usage of curve fitting techniques for modeling data. It is not intended as professional advice for scientific or engineering applications.

# Brief Description:
# The script performs curve fitting on noisy data using both the curve_fit function from scipy.optimize and the lmfit package. It generates noisy data, defines a model function, fits the model to the data using both methods, and then visualizes the results by plotting the original data alongside the fitted curves.

#Author: James Sawyer

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.optimize import curve_fit

# Generate some noisy data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 3 * np.sin(1.5 * x) + np.random.normal(scale=0.3, size=x.size)

# Define a model function
def model_function(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset

# Using curve_fit
popt, pcov = curve_fit(model_function, x, y, p0=[2, 1, 0, 0])

# Using lmfit
model = Model(model_function)
params = model.make_params(amplitude=2, frequency=1, phase=0, offset=0)
result = model.fit(y, params, x=x)

# Plotting
plt.scatter(x, y, label='Data')
plt.plot(x, model_function(x, *popt), label='curve_fit', linestyle='--')
plt.plot(x, result.best_fit, label='lmfit')
plt.legend()
plt.show()
