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

from base import Base


# ================================================================================
# Main class
# ================================================================================
class PCA(Base):
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

    # --------------------------------------------------------------------------------
    # Logic
    # --------------------------------------------------------------------------------
    def get_eigen_values_and_vectors(self, X_meaned: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get eigen values and vectors from the covariance matrix of X
        [Note]
        C: Covariance matrix
        V: Eigen vector in column based matrix where each column is an eigen vector.
        V(-1): Inverse of V
        D: Eigen value diagonal matrix where each diagonal is an eigen value
        C = VDV(-1)

        https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
        w: eigen values
        v: eigenvectors, such that the column v[:,i] is the eigenvector foreigenvalue w[i].

        Numpy eig returns V where each column is an eigen vector.

        Args:
            X_meaned: meaned ata points, (X - u) the covariance matrix of which to extract eigen values and vectors
        Returns: (eigen_values, eigen_vectors)
        """
        name: str = "get_eigen_values_and_vectors()"
        values, vectors = np.linalg.eig(np.cov(m=X_meaned, rowvar=False))

        self.logger.debug("%s: eigen vectors %s", name, vectors[:5])
        self.logger.debug("%s: values=%s", name, values[:5])
        return values, vectors

    def sort_eigen_vectors(
            self,
            eigen_vectors: np.ndarray,
            eigen_values: np.ndarray
    ) -> np.ndarray:
        """Sort Eigen vectors per descending eigen values
        Args:
            eigen_vectors:
            eigen_values
        Returns: sorted eigen vectors (principal component first)
        """
        name: str = "sort_eigen_vectors()"
        descending_sorted_eigen_value_indices: np.ndarray = (-eigen_values).argsort()
        _sorted: np.ndarray = eigen_vectors[
            :,
            descending_sorted_eigen_value_indices       # Columns = eigen vectors
        ]

        self.logger.debug("%s: eigen values = %s", name, eigen_values)
        self.logger.debug("%s: sorted indices = %s", name, descending_sorted_eigen_value_indices)

        return _sorted

    def num_principal_components_to_cover_threshold(
            self,
            eigen_values: np.ndarray,
            threshold: float
    ) -> np.ndarray:
        """Get the number of principal components that covers the threshold.
        Args:
            eigen_values:
            threshold:
        Returns:
            Number of principal components that preserves threshold of the information
        """
        assert len(eigen_values > 0), f"invalid eigen_values {eigen_values}"

        name: str = "num_principal_components_to_cover_threshold()"
        num_principal_components: np.ndarray = -np.inf

        # Cumulative sums on descending sorted eigen values
        cumulatives: np.ndarray = np.cumsum(np.sort(eigen_values)[::-1]) / np.sum(eigen_values)
        self.logger.debug(
            "%s: cumulatives of eigen values (normalized) %s",
            name, cumulatives
        )

        # Fine the index that exceeds the threshold
        num_principal_components = np.min(np.argwhere(cumulatives >= threshold)) + 1
        assert 0 < num_principal_components <= len(eigen_values), \
            f"invalid num_principal_components {num_principal_components}"
        self.logger.debug(
            "%s: number of principal components to cover thresholds [%s] is [%s].",
            name, threshold, num_principal_components
        )

        return num_principal_components

    def fit(self, X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Train the model to fit the data X
        Args:
            X: Data to reduce the dimension
            threshold:
                Percentage of the information to preserve.
                Default 0.95 for 95% of the original X information
        Returns:
            Dimension reduced data
        """
        assert X is not None and len(X) > 0, f"invalid X: {X}"

        X_reduced: np.ndarray = np.inf
        try:
            # Get eigen vector and values
            X_meaned: np.ndarray = (X - np.mean(X, axis=0))
            eigen_values, eigen_vectors = self.get_eigen_values_and_vectors(X_meaned=X_meaned)
            self.logger.debug(
                "top 10 eigen values: %s", eigen_values[:10] if len(eigen_values) > 10 else eigen_values
            )
            sorted_eigen_vectors: np.ndarray = self.sort_eigen_vectors(
                eigen_values=eigen_values, eigen_vectors=eigen_vectors
            )
            # Extract the eigen vectors using the indices covering the PC.
            num_principal_components = self.num_principal_components_to_cover_threshold(
                eigen_values=eigen_values, threshold=threshold
            )
            principal_eigen_vectors = sorted_eigen_vectors[
                :,
                0: num_principal_components     # Each column is eigen vectors
            ]
            # d: dimension of X (and eigen vector)
            # n: number of rows in X
            # k: number of principal components
            X_reduced = np.einsum("nd,dk->nk", X_meaned, principal_eigen_vectors)

        except (KeyError, IndexError) as e:
            self.logger.debug("PCA failed with %s", str(e))
            raise RuntimeError("PCA failed") from e

        finally:
            pass

        assert len(X_reduced) > 0, f"Invalid result {X_reduced}"

        self.logger.info("Reduced X shape %s", X_reduced.shape)
        return X_reduced


# ================================================================================
# Tests (to be separated)
# ================================================================================
def test_fit():
    n_components: int = 3
    import matplotlib.pyplot as plt

    # unused but required import for doing 3d projections with matplotlib < 3.2
    import mpl_toolkits.mplot3d  # noqa: F401

    from sklearn import datasets

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    pca = PCA()
    X_reduced = pca.fit(X=iris.data, threshold=1.0)[
        :,
        0:n_components
    ]

    from sklearn.decomposition import PCA as spca
    s_reduced = spca(n_components=n_components).fit_transform(iris.data)
    assert np.allclose(a=np.abs(s_reduced), b=np.abs(X_reduced), atol=1e-05), \
        f"invalid PCA result. expected:\n {s_reduced[0:10, :]}\nactual:\n{X_reduced[0:10, :]}"

    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()

    assert True
