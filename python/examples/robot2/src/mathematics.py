from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    TypedDict,
    Final
)
import numpy as np


def rotation_matrix(theta: int) -> np.ndarray:
    """Rotate the vector with theta degree clock-wise
    Args:
        theta: degrees to rotate
    Return:
        rotation matrix
    """
    radian = np.radians(theta)
    return np.array([
        [np.cos(radian), np.sin(radian)],
        [-np.sin(radian), np.cos(radian)]
    ])


def rotate_vector(vector: Union[List[int], np.ndarray], theta: int) -> Union[List[int], np.ndarray]:
    """Rotate the vector with theta degree clock-wise
    Args:
        vector: vector to rotate
        theta: degrees to rotate
    Return:
        rotated vector
    """
    rotation = rotation_matrix(theta)
    rotated = rotation.dot(vector).astype(int)
    return rotated
