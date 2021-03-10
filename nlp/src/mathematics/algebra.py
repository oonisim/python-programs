import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=120)


def rotate(X, θ):
    """Rotate X with θ radians counter-clockwise
    Args:
        X: data of shape (N, 2) where each point is (x, y) and N is number of points.
        θ: radian to rotate
    Returns:
        Z: roated data of shape (N, 2)
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    Z = np.matmul(
        np.array([
            [np.cos(θ), -np.sin(θ)],
            [np.sin(θ), np.cos(θ)]
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

def is_point_inside_sector(X: np.ndarray, base: float, coverage: float):
    """
    Check if the point(x, y) is within or on the sector with the coverage angle
    starting from the base angle. For instance, base=1/4 pi and coverage is pi,
    then if a point is on/between angles [1/4pi, 5/4pi], returns true.

    Domain: 0 <= base <2pi and 0 <= coverage < 2pi
        (base, coverage)=(b, 0) checks if a point is on the angle b.
        (base, coverage)=(b, 2pi) is invalid as it accepts any points.


    If coverage > 2pi,

    Args:
        X: Points of (x, y) coordinates. Shape (N,2)
            x is x-coordinate of a poit
            y is y-coordinate of a point
        base: base angle to start the coverage in radian units
        coverage: angles to cover in radian units.
    Returns:
        Y: Labels of shape (N,). True if (x,y) is on or between the coverage.
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=float).reshape(-1, 2)

    assert (0 <= base < 2 * np.pi) and (0 <= coverage < 2 * np.pi), \
        "base and coverage need [0, 2pi) but base %s, coverage %s" \
        % (base, coverage)
    end = base + coverage

    # y: shape(N,)
    y = X[
        ::,
        1
        ]
    # x: shape(N,)
    x = X[
        ::,
        0
        ]

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

    return Y
