import numpy as np
import sys
import matplotlib.pyplot as plt

COLOR_LABELS = np.array([
    'red',
    'green',
    'blue',
    'pink',
    'violet',
    'cyan',
    'magenta',
    'azure',
    'orange',
    'grey',
    'salmon',
    'gold',
    'lime',
    'slategrey',
    'olive',
    'crimson',
    'navy',
    'lavender',
    'yellow',
    'deeppink'
])


def plot_categorical_predictions(axes, grid, X, Y, predictions):
    """
    Args:
        axes: matplot axes
        grid: [x_grid, y_grid]
        X: data
        Y: color labels for each category
        predictions: predictions from the model
    """
    # Plot data and labels
    # axes.scatter(X[:, 1], X[:, 2], c=T, s=40, cmap=plt.cm.Spectral)
    axes.scatter(X[:, 1], X[:, 2], c=Y, s=40)
    # plot predictions
    axes.contourf(grid[0], grid[1], predictions, cmap=plt.cm.Spectral, alpha=0.3)

    axes.set_xlim(grid[0].min(), grid[0].max())
    axes.set_ylim(grid[1].min(), grid[1].max())
    axes.grid()

