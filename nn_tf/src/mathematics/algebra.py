import sys
import numpy as np
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL
)


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=120)


def rotate(X, radian):
    """Rotate X with radians counter-clockwise
    Args:
        X: data of shape (N, 2) where each point is (x, y) and N is number of points.
        radian: radian to rotate
    Returns:
        Z: rotated data of shape (N, 2)
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    Z = np.matmul(
        np.array([
            [np.cos(radian), -np.sin(radian)],
            [np.sin(radian), np.cos(radian)]
        ]),
        X.T     # (N, 2) to (2, M) to be able to (2, 2) @ (2, N)
    )
    return Z.T


def shift(X, offsets):
    """Shift X offset
    Args:
        X: data of shape (N, D)
        offsets: distance to offset. shape (D) or (1, D), or (N, D)
    Returns:
        Z: shifted X
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(offsets, np.ndarray):
        offsets = np.array(offsets)

    if X.ndim == offsets.ndim:
        return X + offsets
    elif X.ndim == offsets.ndim + 1:
        offsets = offsets[
            np.newaxis,
            ::
        ]
        return X + offsets
    else:
        assert False


def is_point_inside_sector(X: np.ndarray, base: float, coverage: float, centre=None, radius: float = -1.0):
    """
    Check if the point(x, y) is within or on the sector with the coverage angle
    starting from the base angle. For instance, base=1/4 pi and coverage is pi,
    then if a point is on/between angles [1/4pi, 5/4pi], returns true.

    Domain: 0 <= base <2pi and 0 <= coverage < 2pi
        (base, coverage)=(b, 0) checks if a point is on the angle b.
        (base, coverage)=(b, 2pi) is valid to check if a point is in a circle


    If coverage > 2pi,

    Args:
        X: Points of (x, y) coordinates. Shape (N,2)
            x is x-coordinate of a poit
            y is y-coordinate of a point
        base: base angle to start the coverage in radian units
        coverage: angles to cover in radian units.
        centre: if not None, the centre of the sector
        radius: limit of the sector when > 0. When <=0, no check
    Returns:
        Y: Labels of shape (N,). True if (x,y) is on or between the coverage.
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=TYPE_FLOAT).reshape(-1, 2)

    assert (0 <= base < 2 * np.pi) and (0 <= coverage <= 2 * np.pi), \
        "base and coverage need [0, 2pi) but base %s, coverage %s" \
        % (base, coverage)
    end = base + coverage

    # y: shape(N,)
    X = np.copy(X)
    y = X[
        ::,
        1
    ]
    # x: shape(N,)
    x = X[
        ::,
        0
    ]

    if centre is not None and isinstance(centre, np.ndarray) and centre.size == 2:
        centre = centre.reshape(2)
        x -= centre[0]
        y -= centre[1]

    # --------------------------------------------------------------------------------
    # angle=atan2(y/x) is preserved if angle > 0 or mapped to the copmplement
    # of 2pi via (2pi - angle) so that the angle is on/between 0 and 2*pi.
    # angle:shape(N,)
    # --------------------------------------------------------------------------------
    angle = (np.arctan2(y, x) + (2 * np.pi)) % (2 * np.pi)

    # --------------------------------------------------------------------------------
    # Y:shape(N,) = AND[ (angle >= base):(N,), (angle < base):(N,) ]
    # Inter-array element-wise AND between array(angle >= base) and array(angle < base).
    # --------------------------------------------------------------------------------
    Y = np.logical_and((angle >= base), (angle < end))

    # --------------------------------------------------------------------------------
    # Check if a point is inside the radius.
    # --------------------------------------------------------------------------------
    if radius > 0:
        radii = np.sqrt(np.power(x, 2) + np.power(y, 2))
        Y = np.logical_and(Y, (radii <= radius))

    return Y


def test():
    X = np.array([
        [-0.5, 0],
        [-2, -3]
    ])
    centre = (1, 1)
    X = X[
        is_point_inside_sector(
            X=X,
            base=0,
            coverage=2 * np.pi,
            centre=centre,
            radius = np.sqrt(5)
        )
    ]
    print(X)
