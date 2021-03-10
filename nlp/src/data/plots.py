"""Data for classifications"""
import numpy as np


def set_in_a_radius(radius: float, d: int, n: int):
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
    ).astype(float)
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
