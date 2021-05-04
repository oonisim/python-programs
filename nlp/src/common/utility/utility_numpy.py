import numpy as np


def random_bool_tensor(shape: tuple, num_trues: int):
    """Generate bool tensor where num_trues elements are set to True
    Args:
        shape: shape of the tensor to generate
        num_trues: number of True to randomly set to the tensor
    Returns: tensor of shape where num_trues elements are set to True
    """
    size = np.multiply.reduce(array=shape, axis=None)   # multiply.reduce(([])) -> 1
    assert len(shape) > 0 <= num_trues <= size

    indices = np.random.choice(a=np.arange(size), size=num_trues, replace=False)
    flatten = np.zeros(size)
    flatten[indices] = 1

    return np.reshape(flatten, shape).astype(np.bool_)
