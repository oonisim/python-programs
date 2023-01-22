import numpy as np

from util_constant import (
    TYPE_FLOAT,
    TYPE_INT,
)
from util_numpy import (
    get_cosine_similarity,
    get_orthogonal_3d_vectors,
    get_orthogonal_vectors,
    is_all_close
)


def test_get_orthogonal_vectors__success():
    """
    Verify the dot products of the orthogonal vectors are 0
    """
    x, y = get_orthogonal_3d_vectors()
    is_all_close(np.dot(x, y), TYPE_FLOAT(0))

    dimension: int = np.random.randint(2, 10)
    Z = get_orthogonal_vectors(dimension=dimension)
    chosen: np.ndarray = np.random.choice(range(dimension), size=2, replace=False)
    x_index = chosen[0]
    y_index = chosen[1]
    is_all_close(np.dot(Z[x_index, :], Z[y_index, :]), TYPE_FLOAT(0))


def test_get_cosine_similarity_fail():
    """
    Test conditions
        1: argument x or y as array with dimension 0 fails.
        2: Similarities between orthogonal vector itself result in 1
    """
    # --------------------------------------------------------------------------------
    # Test condition #1: argument x or y as array with dimension 0 fails.
    # --------------------------------------------------------------------------------
    x: np.ndarray = np.array([0, np.random.normal()])
    y: np.ndarray = np.array(1)
    try:
        get_cosine_similarity(x=x, y=y)
        get_cosine_similarity(x=y, y=x)
    except AssertionError as e:
        pass
    else:
        assert False, f"scalar argument y [{y} should fail get_cosine_similarity()"

    x: np.ndarray = np.array([0, np.random.normal()])
    y: np.ndarray = np.array([[1]])
    try:
        get_cosine_similarity(x=x, y=y)
        get_cosine_similarity(x=y, y=x)
    except AssertionError as e:
        pass
    else:
        assert False, f"scalar argument y [{y} should fail get_cosine_similarity()"


def test_get_cosine_similarity_success():
    """
    Test conditions
        1: Similarities among orthogonal vectors result in 0
        2: Similarities between orthogonal vector itself result in 1
    """
    # --------------------------------------------------------------------------------
    # Test condition #1: Orthogonal vector cosine similarity is 0.
    # --------------------------------------------------------------------------------
    # Test similarity between x, y where x and y are scalars
    x: np.ndarray = np.array([0, np.random.normal()])
    y: np.ndarray = np.array([np.random.normal(), 0])
    assert is_all_close(get_cosine_similarity(x, y), TYPE_FLOAT(0)), \
        f"expected cosine similarity 0 between orthonormal vectors [{x}] and [{y}], " \
        f"got [{get_cosine_similarity(x, y)}]"

    # Generate matrix Z in shape (D, D) where its row is an orthogonal vector.
    # Split Z to X, Y, then cosine similarity between X, Y should be 0.
    dimension: int = np.random.randint(2, 10)
    index_to_split_at: int = np.random.choice(range(1, dimension))

    Z: np.ndarray = get_orthogonal_vectors(dimension=dimension)
    X: np.ndarray = Z[:index_to_split_at]
    Y: np.ndarray = Z[index_to_split_at:]
    assert len(X) + len(Y) == len(Z)

    similarities: np.ndarray = get_cosine_similarity(X, Y)
    assert is_all_close(similarities, np.zeros(shape=similarities.shape)), \
        f"expected cosine similarity 0 between orthonormal vectors \n{X}\n and \n{Y}\n, " \
        f"got \n{similarities}"

    # --------------------------------------------------------------------------------
    # Test condition #2: Same vector cosine similarity is 1.
    # --------------------------------------------------------------------------------
    assert is_all_close(get_cosine_similarity(x, x), TYPE_FLOAT(1)), \
        f"expected cosine similarity between itself [{x} got [{get_cosine_similarity(x, x)}]"

    # Cosine similarity between itself is at the diagonal only.
    # Non diagonal is dot(X[i:], X[k:]) which will be 0.
    similarities: np.ndarray = np.diag(get_cosine_similarity(Y, Y))
    assert is_all_close(similarities, np.ones(shape=similarities.shape)), \
        f"expected cosine similarity 1 between same vector itself \n{Y}\n" \
        f"got \n{similarities}"
