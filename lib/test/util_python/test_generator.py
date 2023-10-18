"""
Test module for generator utilities
"""
import random
from util_python.generator import (     # pylint: disable=import-error
    split,
    batch
)
import numpy as np
import pandas as pd


def test_batch_fail_with_invalid_data():
    """Test batch fail with invalid data
    Test conditions:
    1. no data to stream.
    2. data is not slice-able e.g. dictionary
    3. batch_size <= 0
    """
    try:
        # Test condition 1
        batch(sliceable=None, batch_size=1)
        batch([], batch_size=1)

        # Test condition 2
        batch(sliceable={1, 2, 3}, batch_size=1)

        # Test condition 3
        batch(range(1), batch_size=0)
        assert False, "expected batch() fails."
    except (AssertionError, RuntimeError):
        pass


def test_batch_succeed():
    """Test batch succeeds with valida data
    Test conditions
    1. data of length 1
    2. data of length N > 1
    """
    # Test condition 1 & 2
    # Test condition 2

    for index in range(1, 1000):
        sequence = range(random.randint(1, index))
        batch_size: int = random.randint(1, index)

        assert sum(batch(sliceable=tuple(sequence), batch_size=batch_size), ()) == tuple(sequence)
        assert sum(batch(sliceable=list(sequence), batch_size=batch_size), []) == list(sequence)

        array = np.array(list(sequence))
        assert np.array_equal(
            np.concatenate(list(batch(sliceable=array, batch_size=batch_size))),
            array
        )
        df = pd.DataFrame({"column": list(sequence)}, index=list(sequence))
        assert df.equals(pd.concat(batch(sliceable=df, batch_size=batch_size)))


def test_split_succeed():
    """Test split succeeds with valida data
    Test conditions
    1. data of length 1
    2. data of length N > 1
    """
    # Test condition 1 & 2
    # Test condition 2

    for index in range(1, 1000):
        sequence = range(random.randint(1, index))
        num_batches: int = random.randint(1, index)

        assert sum(split(sliceable=list(sequence), num_batches=num_batches), []) == list(sequence)

        array = np.array(list(sequence))
        assert np.array_equal(
            np.concatenate(list(split(sliceable=array, num_batches=num_batches))),
            array
        )
        df = pd.DataFrame({"column": list(sequence)}, index=list(sequence))
        assert df.equals(pd.concat(split(sliceable=df, num_batches=num_batches)))
