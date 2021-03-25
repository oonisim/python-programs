"""Data for classifications"""
from typing import (
    Tuple,
    List
)
import numpy as np
from common.constants import (
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_LABEL
)
from mathematics import (
    rotate,
    shift,
    is_point_inside_sector
)


def set_in_a_radius(radius: TYPE_FLOAT, d: int, n: int):
    """Generate cartesian coordinate points in a radius in a D dimension space.
    Args:
        radius: A distance within which data points are to be generated
        d: dimensions
        n: number of points to generate
    Returns:
        cartesians: data points in cartesian coordinate of shame (N,D)

    Mapping from n-spherical coordinate to a Cartesian coordinate.
    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    https://math.stackexchange.com/questions/991169

    Unlearn the (x, y, z) coordinates and work on according to the dimensions.
    In a n-spherical coordinate, a point P = (radius, Θ1, Θ2, ... Θi, ..., Θd-1).
    Θi is an angle form axis i to the vector P. Then the xi in a Cartesian system
    xi = cos(Θi) * sin(i+1) * ... * sin(Θd-2) * sin(Θd-1) when i > 0.
    xi = sin(Θi) * sin(i+1) * ... * sin(Θd-2) * sin(Θd-1) when i = 0.
    """
    assert n > 0 and d > 0, radius > 0.0

    # --------------------------------------------------------------------------------
    # Radii with one radius per each data point.
    # Radians for (d - 1) dimensions in a data point (minus 1 for radius dimension)
    # --------------------------------------------------------------------------------
    radii: np.ndarray = np.random.uniform(0, radius, n).reshape((n, 1))
    radians: np.ndarray = np.random.uniform(0, 2 * np.pi, (n, d - 1))

    # --------------------------------------------------------------------------------
    # numpy universal function to generate sin(Θi) * sin(i+1) * ... * sin(Θd-1) at i,
    # then accumulate them for each i.
    # --------------------------------------------------------------------------------
    def func(left, right):
        return left * right

    ufunc = np.frompyfunc(func, 2, 1)

    # --------------------------------------------------------------------------------
    # Generate sin(Θd-1) * ... * sin(i+1) * sin(Θi) for each i in (1, 2, ... d-1).
    # Add ones at column 0, which is later placed at the last column (d-1) in (N,D),
    # after reversed, then becomes cos(Θd-1) as the cartesian coordinate X[(d-1)].
    #
    # Apply ufunc to radians:(N, D-1) without reversing along axis=-1 because there is
    # no-order in the array "radians" before apply ufunc as they are random.
    # Reverse the result sin(Θi) * ... sin(i+1) * sin(Θd-1) along the axis -1 to form
    # [ sin(Θ1) * ... * sin(Θd-1), ..., sin(Θi) * ... * sin(Θd-1), ..., sin(Θd-1), 1 ].
    # --------------------------------------------------------------------------------
    cartesians = np.c_[np.ones(n), np.sin(radians)]
    ufunc.accumulate(
        cartesians,
        axis=-1,
        dtype=np.ndarray,
        out=cartesians
    ).astype(TYPE_FLOAT)
    cartesians = np.flip(cartesians, axis=-1)

    # --------------------------------------------------------------------------------
    # cos(Θi) to generate xi = cos(Θi) * sin(i+1) * ... * sin(Θd-1) for i > 0.
    # Ones at d=0 so that x0 = sin(Θ1) * sin(Θ1 ) * ... * sin(Θd-1) for i = 0.
    # cosins = [1, cos(Θ1), cos(Θ2), ... cos(Θd-1)]
    # --------------------------------------------------------------------------------
    cosines = np.c_[np.ones(n), np.cos(radians)]

    # --------------------------------------------------------------------------------
    # Element multiply cosines [1, cos(Θ1), cos(Θ2), ... cos(Θd-1)] with
    # [ sin(Θ1) * ... * sin(Θd-1), ..., sin(Θi) * ... * sin(Θd-1), ..., sin(Θd-1), 1 ]
    # generates:
    # cos(Θi) * sin(i+1) * ... * sin(Θd-2) * sin(Θd-1) when i > 1.
    # sin(Θi) * sin(i+1) * ... * sin(Θd-2) * sin(Θd-1) when i = 0.
    # --------------------------------------------------------------------------------
    np.multiply(cosines, cartesians, out=cartesians)
    np.multiply(radii, cartesians, out=cartesians)

    del radii, cosines

    return cartesians


def sets_in_circles(
        radius: TYPE_FLOAT,
        ratio: TYPE_FLOAT = 1.0,
        m: int = 3,
        n: int = 10000
):
    """Generate m set of coordinates where each set forms plots in a circle
    Args:
        radius: circle radius
        ratio: how far to locate a new centre of a circle
        m: number of circles
        n: Number of points in a circle
    Returns:
        circles: coordinates of circles. Shape (m, n, d). (n, d) per circle
        centre: coordinates of the centre of each circle. Shape (m, 2)
    """
    assert 2 <= m <= n and ratio > 0 and radius > 0.0

    radius = TYPE_FLOAT(radius)
    d = 2   # circle is in 2D
    circle = set_in_a_radius(radius=radius, d=d, n=n)

    # Generate a new circle by shifting the centre of the "circle" to a "centre".
    # The coordinate of the new centre = rotate(base, step * i).
    base = np.array([radius * ratio, TYPE_FLOAT(0)])
    step = (2 * np.pi) / (m-1)
    step = step if step < (np.pi / 2) else np.pi / m

    def __rotate(angle):
        return rotate(X=base, radian=angle)

    centres = list(map(__rotate, [step * i for i in range(0, m-1, 1)]))
    centres.insert(0, np.array([0.0, 0.0]))    # add the original circle
    mean = np.mean(np.array(centres), axis=0)
    centres = np.array([_centre - mean for _centre in centres])

    def __relocate(location):
        return shift(X=circle, offsets=location)

    circles = np.array(list(map(__relocate, centres)))
    assert circles.shape == (m, n, d), f"{circles.shape}\n{circles}"
    assert centres.shape == (m, 2)

    return circles, centres


def set_in_circle_A_and_B(A, centre_B, radius_B):
    # --------------------------------------------------------------------------------
    # Note A_NOT_B can be empty.
    # --------------------------------------------------------------------------------
    A_AND_B = A[
        is_point_inside_sector(
            X=A,
            base=0.0,
            coverage=2 * np.pi,
            centre=centre_B,
            radius=radius_B
        )
    ]
    return A_AND_B


def set_in_circle_A_not_B(A, centre_B, radius_B):
    # --------------------------------------------------------------------------------
    # Note A_NOT_B can be empty.
    # --------------------------------------------------------------------------------
    A_NOT_B = A[
        np.logical_not(
            is_point_inside_sector(
                X=A,
                base=0.0,
                coverage=2 * np.pi,
                centre=centre_B,
                radius=radius_B
            )
        )
    ]
    return A_NOT_B


def sets_of_circle_A_not_B(
        radius: TYPE_FLOAT,
        ratio: TYPE_FLOAT = 1.0,
        m: int = 3,
        d: int = 2,
        n: int = 10000
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Generate m sets of (A NOT B).
        First generate m circles. Let circles[i] be A and circles[i+1] as B.
        A is a set of points that form a circle. B as well. Then A_NOT_B is
        those point in A but not in B.

    Args:
        radius: circle radius
        ratio: how far to locate a new centre of a circle
        d: dimension. 2 only for now
        m: number of circles
        n: Number of points in a circle
    Returns:
        [A_NOT_B]: List of A_NOT_B
        centre: coordinates of the centre of each circle. Shape (m, 2)
        intersection: intersection of all circles.
    """
    assert 2 <= m <= n and ratio > 0 and radius > 0.0
    circles, centres = sets_in_circles(radius=radius, ratio=ratio, m=m, n=n)
    result = []

    assert d == 2, "Only 2 is supported for now"

    intersection = circles[0]
    for i in range(1, m):
        centre_B = centres[i]
        intersection = set_in_circle_A_and_B(intersection, centre_B, radius)
        if intersection.size <= 0:
            break

    for i in range(m):
        A = circles[i]
        centre_B = centres[(i+1) % m]
        A_NOT_B = set_in_circle_A_not_B(A, centre_B, radius)
        result.append(A_NOT_B)

    # ------------------------------------------------------------
    # You cannot do this because the number of rows are different.
    # numpy array a, b, c can exist in a higher dimension only when
    # their shapes are the same.
    #
    # a = np.array([
    #     [-0.68595   , -0.59704907],
    #     [-0.70704997, -0.30369686]])
    # b = np.array([
    #     [ 0.51405   , -0.59704907],
    #     [ 1.51991102, -0.48112732],
    #     [ 0.49295003, -0.30369686]])
    # c = np.array([
    #     [ 0.91991102,  0.55810317],
    #     [-0.10704997,  0.73553363]])
    #
    # d = [a, b, c]
    # np.array(d)
    # ---
    # [Result]
    # array([array([[-0.68595   , -0.59704907],
    #        [-0.70704997, -0.30369686]]),
    #        array([[ 0.51405   , -0.59704907],
    #        [ 1.51991102, -0.48112732],
    #        [ 0.49295003, -0.30369686]]),
    #        array([[ 0.91991102,  0.55810317],
    #        [-0.10704997,  0.73553363]])], dtype=object)
    #
    # With warning:
    # Creating an ndarray from ragged nested sequences (which is a
    # list-or-tuple of lists-or-tuples-or ndarrays with different
    # lengths or shapes) is deprecated. If you meant to do this,
    # you must specify 'dtype=object' when creating the ndarray
    #
    # Hence the only way to return multiple arrays of shape
    # a:(4, M), b:(10, M), c:(7,M) is as a list [a,b,c].
    # ------------------------------------------------------------
    return result, centres, intersection


def venn_of_circle_a_not_b(
        radius: TYPE_FLOAT,
        ratio: TYPE_FLOAT = 1.0,
        m: int = 3,
        n: int = 1000
):
    """
    Args:
        radius: radius of the circle in which to generate data points.
        ratio: Distance between centres as radius * ratio.
        m: number of circles (classes to classify)
        n: number of data points per circle
    """
    circles, centres, intersection = \
        sets_of_circle_A_not_B(radius=radius, ratio=ratio, m=m, n=n)

    X = np.vstack(
        [circles[i] for i in range(m-1)] +
        [intersection]
    ).astype(TYPE_FLOAT)

    T = np.hstack(
        [np.full(circles[i].shape[0], i) for i in range(m-1)] +
        [np.full(intersection.shape[0], m-1)]
    ).astype(TYPE_INT)
    assert T.shape[0] == X.shape[0]

    return X, T


def spiral(k: int, d: int = 2, m: int=3):
    """Generate spiral data points
    https://cs231n.github.io/neural-networks-case-study/#data
    Args:
        k: number of points per class
        d: dimensionality
        m: number of classes
    Returns:
        X: spiral data coordinates of shape (n, d)
        y: label (0, 1, ...m-1)
    """
    assert k >= m > 1
    assert d == 2, "currently only d==3 for 2D (bias, x1, x2) is valid"

    N = k
    D = d
    M = m
    X = np.zeros((N * M, D))
    y = np.zeros(N * M, dtype='uint8')
    for j in range(M):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y
