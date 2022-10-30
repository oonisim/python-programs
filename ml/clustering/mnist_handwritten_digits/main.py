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
    # pylint disable=too-many-instance-attributes
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
    def normalize(self, X: np.ndarray):
        """(X - X_mean) / STD(X)"""
        return (X - X.mean(axis=0)) / np.std(X, axis=0)

    def __init__(self):
        super().__init__()
        # Digit data
        self.data, self.labels, self.n_labels = load_digit_data()
        self.N = len(self.data)     # Number of data points
        self.K = self.n_labels      # Number of centroids (K means)

        # Number of principal components to use to represent a digit
        self.D: int = 2             # Dimension=2

        # PCA reduced data of digits into 2 dimensional (PCA0, PCA1)
        self.X: np.ndarray = self.normalize(pca_reduce_data(data=self.data, n_components=self.D))

        # Set initial centroids from X at random
        self.centroids: np.ndarray = self.X[np.random.choice(self.N, self.n_labels)]

        # Number of training to run
        self.epoch: int = 1000

    # --------------------------------------------------------------------------------
    # Logic
    # --------------------------------------------------------------------------------
    def update_centroids(self):
        # --------------------------------------------------------------------------------
        # Cluster the digits deta in two principal components space with n_labels centroids
        # --------------------------------------------------------------------------------

        # Calculate the distance from each x to centroids (each (x, c(n)) pair) -> row wise sum (axis=-1)
        # (X:(N, D) - centroids:(K, D))**2.sum(axis=-1)
        distances: np.ndarray = np.square(      # Shape (N, D)
            self.X[                             # Shape (N, 1, D) for broadcast
                :,
                np.newaxis,
                :
            ] - self.centroids                  # (K, D) -> (1, K, D) broadcast
        ).sum(axis=-1)                          # (N, K, D) -> (N, K) reduce
        assert distances.shape == (self.N, self.K), \
            f"distances.shape expected {(self.N, self.K)} got {distances.shape}"

        # Find the nearest centroid for each x -> row-wise argmin
        nearest_centroid_id_to_x: np.ndarray = distances.argmin(axis=-1)

        # Get a cluster of x for each centroid
        clusters: List[np.ndarray] = []
        for cluster_id in range(0, self.K):
            # Extract the group of x nearest to the current centroid
            cluster: np.ndarray = self.X[
                nearest_centroid_id_to_x == cluster_id
            ]
            # Cluster shape is (num_x_in_cluster, D)
            num_x_in_cluster: np.ndarray = (nearest_centroid_id_to_x == cluster_id).astype(int).sum()
            assert cluster.shape == (num_x_in_cluster, self.D), \
                f"cluster.shape expected {(num_x_in_cluster, self.D)} got {cluster.shape}."

            clusters.append(cluster)

        # Update the centroid coordinate for each cluster identified
        for cluster_id in range(len(clusters)):
            # column-wise mean of the x in the cluster is the new centroid
            self.centroids[cluster_id] = clusters[cluster_id].mean(axis=0)
            assert self.centroids.shape == (self.K, self.D)

    def fit_transform(self):
        try:
            for _ in range(self.epoch):
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
    plt.ylabel("PCA2")

    # Plot centroids
    for i in range(len(centroids)):
        # label = TextPath((0,0), str(i))
        # [marker thickness] https://stackoverflow.com/questions/17285163/
        plt.plot(centroids[i, 0], centroids[i, 1], color="red", marker="x", markersize=20, mew=5)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == "__main__":
    main()
