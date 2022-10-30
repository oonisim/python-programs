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
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def load_digit_data() -> Tuple[np.ndarray, np.ndarray, int]:
    """Load digit data
    """
    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

    return data, labels, n_digits


def pca_reduce_data(data: np.ndarray, n_components: int):
    """Reduce dimension with PCA
    Args:
        data: data to reduce
        n_components: number of principal component to return
    """
    return PCA(n_components=n_components).fit_transform(data)
