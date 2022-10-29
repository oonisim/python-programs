""" Module docstring
[Objective]

[Prerequisites]

[Assumptions]

[Note]

[TODO]

"""
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Union,
    Optional,
)
import os
import sys
import logging

import numpy as np
# unused but required import for doing 3d projections with matplotlib < 3.2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib.text import TextPath

from base import Base
from utility import (
    load_digit_data,
    pca_reduce_data,
)


# ================================================================================
# Main class
# ================================================================================
class KMeans(Base):
    """
    Main class for [TBD]
    """
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Class
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        # Digit data
        self.data, self.labels, self.n_labels = load_digit_data()
        self.N = len(self.data)    # Number of data points

        # Number of principal components to use to represent a digit
        self.D: int = 2         # Dimension=2

        # PCA reduced data of digits
        X: np.ndarray = pca_reduce_data(data=self.data, n_components=self.D)
        X_meaned: np.ndarray = X - X.mean(axis=0)
        self.X: np.ndarray = X_meaned / X.std(axis=0)

        # Choose the initial centroids:(n_labels, D) from X at random.
        self.centroids: np.ndarray = self.X[
            np.random.choice(a=self.N, size=self.n_labels, replace=False),
            :
        ]
        assert self.centroids.shape == (self.n_labels, self.D), \
            f"centroid shape expected {(self.n_labels, self.D)} bot {self.centroids.shape}."

        # Number of training to run
        self.epoch: int = 1000

    # --------------------------------------------------------------------------------
    # Logic
    # --------------------------------------------------------------------------------
    def update_centroids(self):
        # --------------------------------------------------------------------------------
        # Cluster the digits deta in two principal components space with n_labels centroids
        # --------------------------------------------------------------------------------
        # Get Distances to all points form each centroids.
        # For each x:(1, D), get distance from 10 centroids:(10,D).
        # -> distances:(10, D) = ((x:(1, D) - centroids(10, D)) **2).sum(axis=-1)
        #
        # Apply the calculation to N number of x:(1, D).
        # -> Distances:(N, 10) = X:(N, D) and centroids:(10, D)
        # 1. X:(N, D) -> X:(N, 1, D)
        # 2. ((X:(N, 1, D) - centroids:(10, D))**2).sum(axis=-1)
        distances: np.ndarray = ((self.X[:, np.newaxis, :] - self.centroids)**2).sum(axis=-1)
        assert distances.shape == (self.N, self.n_labels), \
            f"distances.shape expected {(self.N, self.n_labels)} got {distances.shape}."

        # List of the closest centroid id to each X
        # closest:(N, 1) = distances.argmin(axis=-1)
        closest_centroid_id_to_x: np.ndarray = distances.argmin(axis=-1)
        assert closest_centroid_id_to_x.shape == (self.N,), \
            f"closest_centroid_id_to_x.shape expected {(self.N,)} got {closest_centroid_id_to_x.shape}."

        # Get 10 x cluster where each cluster is close to a current centroid
        centroid_clusters: List[np.ndarray] = list()
        for cluster_id in range(0, self.n_labels):
            centroid_cluster_selector: np.ndarray = (closest_centroid_id_to_x == cluster_id)
            cluster: np.ndarray = self.X[centroid_cluster_selector]

            num_in_cluster: np.ndarray = centroid_cluster_selector.astype(int).sum()
            assert cluster.shape == (num_in_cluster, self.D), \
                f"cluster.shape expected {(num_in_cluster, self.D)} got {cluster.shape}."
            centroid_clusters.append(cluster)

        # Update the centroids from the cluster
        # New coordinate of a centroid:(1,D) = cluster.mean(axis=0)
        for cluster_id in range(0, self.n_labels):
            self.centroids[cluster_id] = centroid_clusters[cluster_id].mean(axis=0)

    def fit_transform(self):
        try:
            for i in range(self.epoch):
                self.update_centroids()

        except (KeyError, IndexError) as e:
            self.logger.debug("PCA failed with %s", str(e))
            raise RuntimeError("PCA failed") from e

        finally:
            pass

        return self.X, self.centroids, self.labels


# ================================================================================
# Tests (to be separated)
# ================================================================================
def main():
    kmeans = KMeans()
    X, centroids, labels = kmeans.fit_transform()

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    # [color bar]
    # https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
    sc = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Set1, edgecolor="k")
    plt.colorbar(sc)
    plt.xlabel("PCA1")
    plt.ylabel("PCA1")

    # Plot centroids
    for i in range(len(centroids)):
        label = TextPath((0,0), str(i))
        # [marker thickness] https://stackoverflow.com/questions/17285163/
        plt.plot(centroids[i, 0], centroids[i, 1], color="red", marker="x", markersize=20, mew=5)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == "__main__":
    main()