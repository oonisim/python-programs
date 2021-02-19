"""Normalization layer"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Final,
    Generator,
    Iterator,
    Callable
)
import logging
import copy
import numpy as np
from layer import Layer
from optimizer import (
    Optimizer,
    SGD,
)
from common.functions import (
    numerical_jacobian
)