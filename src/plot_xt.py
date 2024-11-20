import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv('data/xt_plot_worldcup.csv')

# meshgrid
X = np.linspace(0, 100, data.shape[1])
Y = np.linspace(0, 60, data.shape[0])
X, Y = np.meshgrid(X, Y)

# contour plot without interpolation
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, data, levels=100, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('World Cup 2022')
plt.show()