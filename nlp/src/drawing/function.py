import sys
from itertools import product, combinations
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def plot(X, Y, label=None, color=None, title=None, figsize=(5, 4), xlabel=None, ylabel=None, scale=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel) if xlabel else ...
    ax.set_ylabel(ylabel) if ylabel else ...
    ax.set_title(title) if title is not None else ...
    ax.plot(X, Y, color=color, label=label)
    ax.grid()
    ax.legend()
    ax.set_xscale(scale) if scale else ...
    return fig, ax


def scatter(X, Y, color=None, title=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.axis('equal')
    ax.set_title(title)
    ax.grid()
    ax.scatter(X, Y, color=color)


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
    axes.scatter(X[:, 0], X[:, 1], c=Y, s=40)

    # plot predictions
    # axes.contourf(grid[0], grid[1], predictions, cmap=plt.cm.Spectral, alpha=0.3)
    axes.contourf(grid[0], grid[1], predictions, cmap=plt.cm.gist_rainbow, alpha=0.4, linewidths=3.0)

    axes.set_xlim(grid[0].min(), grid[0].max())
    axes.set_ylim(grid[1].min(), grid[1].max())
    axes.grid()


def draw_sphere(ax, radius, color='indigo'):
    """Draw a sphere in wireframe
    https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html
    https://stackoverflow.com/questions/11140163

    Usage:
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        draw_sphere(ax, radius=2)
    """
    # draw cube
    assert radius > 0
    r = [-radius, radius]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="olive", alpha=0.8)

    # draw sphere wireframe
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color=color, alpha=0.2)
