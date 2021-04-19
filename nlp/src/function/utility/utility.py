from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable,
    Any,
    NoReturn,
    Final
)
import logging
import numpy as np
import numexpr as ne
import tensorflow as tf
from common.constant import (
    TYPE_FLOAT,
    TYPE_LABEL,
    TYPE_TENSOR,
    BOUNDARY_SIGMOID,
    ENABLE_NUMEXPR
)
import function.common.base as base
Logger = logging.getLogger(__name__)


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
        Returns: [
            [b, a, c],
            [c, b, d],
            [d, e, e]
        ] where the first column(s) is event and the rest are context
        """
        length: int = len(sequence)
        stride: int = int((window_size-event_size)/2)

        assert \
            super().is_tensor(sequence) and sequence.ndim == 1 and \
            length >= window_size > event_size > 0, \
            f"Expected a sequence of length >= {window_size} but {sequence}"
        assert (window_size-event_size) % 2 == 0, "Need stride as integer > 0"

        num_windows = length-window_size+1
        context_windows = [
            sequence[slice(index, index+window_size)]
            for index in range(0, num_windows)
        ]

        event_context_paris = np.zeros(shape=(num_windows, window_size))
        # Labels
        event_context_paris[
            ::,
            0: event_size
        ] = context_windows[
            ::,
            event_size:2*event_size
        ]
        # Left contexts
        event_context_paris[
            ::,
            event_size:(event_size+stride)
        ] = context_windows[
            ::,
            0:stride
        ]
        # Right contexts
        event_context_paris[
            ::,
            (event_size + stride):
        ] = context_windows[
            ::,
            (event_size + stride):
        ]

        return event_context_paris

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
