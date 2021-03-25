from layer.constants import *
from layer.base import *
from layer.normalization import *
from layer.matmul import Matmul
from layer.activation import (
    Sigmoid,
    ReLU
)
from layer.objective import *
from layer.activation import *

SCHEME = {
    "batchnormalization": BatchNormalization,
    "matmul": Matmul,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "crossentropylogloss": CrossEntropyLogLoss
}