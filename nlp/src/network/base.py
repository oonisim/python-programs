"""Base for neural network implementations
G=g0(g1(...(gn-1(X))...) = (g0 o ... o gi o ... gn-1)
    Back-propagation passes the gradient dL/dX backwards from the last layer in the network.

"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable,
    Iterable,
    NoReturn,
    Final
)
import logging
import numpy as np
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL
)
from layer.base import Layer


class Network(Layer):
    """Neural network base class"""

    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(
            self,
            name: str,
            num_nodes: int,
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: network ID
            num_nodes: Number of nodes in the first layer
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        assert name
        self._L: Union[TYPE_FLOAT, np.ndarray] = -np.inf
        self._history: List[Union[TYPE_FLOAT, np.ndarray]] = []
        self._history: List[TYPE_FLOAT]
        self._GN: List[Union[TYPE_FLOAT, np.ndarray]] = []   # Numerical gradients GN of layers
        self._dS: List[Union[TYPE_FLOAT, np.ndarray]] = []   # Gradients dL/dS of layers

        # --------------------------------------------------------------------------------
        # Inference layers in the network
        # --------------------------------------------------------------------------------
        self._layer_inference: Layer = None
        self._layer_objective: Layer = None
        self._function: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
        self._predict: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
        self._gradient: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
        self._update: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None

        self._layers_all = List[Layer] = []

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def T(self) -> np.ndarray:
        """Label in OHE or index format"""
        return super().T

    @T.setter
    def T(self, T: Union[np.ndarray, int]):
        self._T = np.array(T, dtype=TYPE_LABEL) if isinstance(T, int) else T.astype(int)
        super(Network, type(self)).T.fset(self, T)

        for __layer in self.layers_all:
            __layer.T = T

    @property
    def L(self) -> Union[TYPE_FLOAT, np.ndarray]:
        """Network objective value (Loss)"""
        assert np.isfinite(self._L), \
            "Objective value L not initialized or an invalid value."
        return self._L

    @property
    def history(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Network objective value (Loss) history"""
        assert self._history, \
            "Objective value L not initialized or an invalid value."
        return self._history

    @property
    def GN(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Numerical gradients of layers"""
        assert self._GN, "Numerical gradients GN of the network not initialized."
        return self._GN

    @property
    def dS(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Gradients dL/dS that have been used to update S in each layer"""
        assert self._dS, "Gradients dL/dS of the network not initialized."
        return self._dS

    @property
    def layer_inference(self) -> Layer:
        """Inference layers"""
        assert isinstance(self._layer_inference, Layer), \
            "Inference layers not initialized"
        return self._layer_inference

    @property
    def layer_objective(self) -> Layer:
        """Objective layers"""
        assert isinstance(self._layer_objective, Layer), \
            "Objective layers not initialized"
        return self._layer_objective

    @property
    def layers_all(self) -> List[Layer]:
        """all layers"""
        assert self._layers_all and len(self._layers_all) > 1, \
            "Network need to have at least 1 objective and 1 inference layer."
        return self._layers_all

    @property
    def num_layers(self) -> int:
        """Number of layers"""
        return len(self._layers_all)

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[TYPE_FLOAT, np.ndarray]) -> Union[TYPE_FLOAT, np.ndarray]:
        """Calculate the network objective (Loss)
        Args:
            X: Batch input
        Returns:
            L: Objective value of the network (Loss)
        """
        X = np.array(X).reshape((1, -1)) if isinstance(X, TYPE_FLOAT) else X
        self._X = np.array(X) if isinstance(X, TYPE_FLOAT) else X
        assert self.X.shape[0] == self.N, \
            f"Batch size of X needs to be {self.N} but {self.X.shape}."
        self._L = self._function(X)
        return self.L

    def gradient_numerical(
            self, h: TYPE_FLOAT = 1e-5
    ) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Numerical Gradients GN"""
        self._GN = [layer.gradient_numerical() for layer in self.layers_all]
        return self.GN

    def gradient(self, dY: Union[np.ndarray, TYPE_FLOAT]) -> Union[TYPE_FLOAT, np.ndarray]:
        """Back propagate gradients"""
        return self._gradient(np.array(1.0))

    def update(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Layer state S updates"""
        __dS = []
        for __layer in self.layers_all:
            __dS += __layer.update()

        self._dS = __dS
        return self.dS

    def validate(self):
        """Validate the gradient G with the numerical gradient GN"""
        assert len(self.dS) == len(self.GN)

        differences: Dict[int, np.ndarray] = {}
        for index, (dS, GN) in enumerate(zip(self.dS, self.GN)):
            deltas = np.array(dS) - np.array(GN)
            avgs = np.average(deltas, axis=0)

            self.logger.debug("--------------------------------------------------------------------------------")
            self.logger.debug("validate: layer[%s] dS=\n%s GN=\n%s", index, dS, GN)
            self.logger.debug("validate: layer[%s] deltas=\n%s avg=\n%s", index, deltas, avgs)
            differences[index] = avgs
            return differences

    def train(self, X: Union[TYPE_FLOAT, np.ndarray], T: Union[TYPE_FLOAT, np.ndarray]):
        self.T = T
        self._L = self.objective(self.function(X))
        self.history.append(self.L)
        self._dX = self.gradient(TYPE_FLOAT(1))
        self.update()
        # D = self.validate()

        print(f"L is network loss is {L}")
        print(f"Gradient dL/dX is {self.dX}")
        print(f"Numerical gradients GN are {self.GN}")
        print(f"Analytical gradients dS are {self.dS}")
