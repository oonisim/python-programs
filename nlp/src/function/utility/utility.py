import string
import random
import logging
import json
import decimal

import numpy as np
import tensorflow as tf

import function.common.base as base
from common.constant import (
    TYPE_TENSOR,
    TYPE_INT
)

Logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if tf.is_tensor(o):
            o = o.numpy()
        if isinstance(o, np.ndarray):
            if o.size == 1:
                o = o.item()
            else:
                o = o.tolist()
        if isinstance(o, decimal.Decimal) or isinstance(o, np.number):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        else:
            return super(DecimalEncoder, self).default(o)


class Function(base.Function):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def event_context_pairs(
            sequence: TYPE_TENSOR,
            window_size: int,
            event_size: int,
    ) -> TYPE_TENSOR:
        """Create (event, context) pairs from a sequence
        For a sequence (a, b, c, d, e) of window_size 3 and event_size 1.
        (event, context) = [
            event=b, context=(a, c),
            event=c, context=(b, d),
            event=d, context=(e, e)
        ]

        Args:
            sequence: 1 dimensional tensor
            window_size: size of the context including the event
            event_size: size of the event which can be > 1

        Returns: (event, context) pairs of shape:(N, E+C) where N=num_windows.
        [
            [b, a, c],
            [c, b, d],
            [d, e, e]
        ] where the first column(s) is event and the rest are context
        """
        length: TYPE_INT = TYPE_INT(len(sequence))
        stride: TYPE_INT = TYPE_INT((window_size-event_size)/2)

        assert \
            super(Function, Function).is_tensor(sequence) and \
            super(Function, Function).tensor_rank(sequence) == 1 and \
            length >= window_size > event_size > 0, \
            f"Expected a sequence of length >= {window_size} but {sequence}"
        assert (window_size-event_size) % 2 == 0, "Need stride as integer > 0"

        # --------------------------------------------------------------------------------
        # The result is a matrix in which windows are stacked up.
        # The result shape is (length-window_size+1, window_size).
        # --------------------------------------------------------------------------------
        num_windows = length-window_size+1
        context_windows = np.array([
            sequence[slice(index, index+window_size)]
            for index in range(0, num_windows)
        ], dtype=TYPE_INT)
        assert context_windows.shape == (num_windows, window_size)
        event_context_paris = np.c_[
            context_windows[    # Labels
                ::,
                slice(stride, (stride + event_size))
            ],
            context_windows[    # Left context
                ::,
                slice(0, stride)
            ],
            context_windows[    # Right context
                ::,
                slice((event_size + stride), None)
            ]
        ]
        assert event_context_paris.shape == (num_windows, window_size)
        return event_context_paris

    @staticmethod
    def random_string(stringLength=8):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))

    @staticmethod
    def pretty_json(dictionary):
        """
        Pretty print Python dictionary
        Args:
            dictionary: Python dictionary
        Returns:
            Pretty JSON
        """
        return json.dumps(dictionary, indent=4, cls=DecimalEncoder)

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: ID name
        """
        super().__init__(name=name, log_level=log_level)
