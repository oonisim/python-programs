"""Encoding utilities"""
import sys
import os
from typing import List, Dict
import numpy as np


def convert_one_hot(indices: np.ndarray, length: int) -> np.ndarray:
    """
    Convert an integer index into one hot encoding (OHE)
    From https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/common/util.py#L73-L94
    Args:
        indices: integer index to convert to OHE (dimension 1 or 2)
        length: length of the one hot encoded array
    Returns: one-hot表現（2次元もしくは3次元のNumPy配列）
    """
    one_hot = None
    N: int = indices.shape[0]

    if indices.ndim == 1:
        one_hot = np.zeros((N, length), dtype=np.int32)
        for idx, word_id in enumerate(indices):
            one_hot[idx, word_id] = 1

    elif indices.ndim == 2:
        C = indices.shape[1]
        one_hot = np.zeros((N, C, length), dtype=np.int32)
        for idx_0, word_ids in enumerate(indices):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


