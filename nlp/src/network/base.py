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
from layer.base import Layer
from common.functions import (
    compose
)


class Base:
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
            layers: List[Layer],
            objective: Layer,
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: network ID
            layers: layers for inference
            objective: objective layer
            log_level: logging level
        """
        assert name
        self._name: str = name
        self._L: Union[float, np.ndarray] = -np.inf     # Network objective value L
        self._X: np.ndarray = np.empty(())              # Bath input to the network
        self._dX: np.ndarray = np.empty(())             # Final gradient
        self._N: int = -1                               # Batch size
        self._T: np.ndarray = np.empty(())              # Labels for the bach.
        self._GN: List[Union[float, np.ndarray]] = []   # Numerical gradients GN of layers
        self._dS: List[Union[float, np.ndarray]] = []   # Gradients dL/dS of layers

        # --------------------------------------------------------------------------------
        # Inference layers in the network
        # --------------------------------------------------------------------------------
        assert layers
        self._layers_inference: List[Layer] = layers
        self.__inference: Callable[[np.ndarray], np.ndarray] = \
            compose(*[layer.forward for layer in layers])

        # Layer i objective function Li: (i for both inference and objective layers)
        # Layer i has its objective function Li = (fn-1 o ... o fi+1) for i < n-1
        # that calculates L=Li(Yi) from its output Yi=fi(Xi). L = Li(fi) i < n-1.
        #
        # Li   = (fn-1 o ... o fi+1)
        # Li-1 = (fn-1 o ... o fi+1 o fi) = Li(fi) = Li o fi
        # L = Li-1(fi-1(Xi-1)) = Li(fi(Yi-1)) where Yi-1=fi-1(Xi-1)
        #
        # Set Li to each inference layer i in the reverse order.
        # Ln-2 for the last inference layer n-2 is fn-1 of the objective layer.
        # Ln-2 = fn-1 by the definition Li = (fn-1 o ... o fi+1).
        Li: Callable[[np.ndarray], np.ndarray] = objective.forward
        for layer in layers[::-1]:
            layer.objective = Li
            # Next Li is Li-1 = Li(fi)
            Li = compose(*[layer.forward, Li])

        # After the first inference layer, Li is the network objective function L.
        # L = (fn-1 o ... o f0)
        self.__objective: Callable[[np.ndarray], np.ndarray] = Li

        # --------------------------------------------------------------------------------
        # Objective layers in the network. List although only one layer.
        # --------------------------------------------------------------------------------
        assert objective
        self._layers_objective: List[Layer] = [objective]

        # There is no Ln-1 for fn-1 of the objective layer. Use an identity as Ln.
        def identity(x: np.ndarray): return x
        objective.objective = identity

        # --------------------------------------------------------------------------------
        # Entire network layers
        # --------------------------------------------------------------------------------
        self._layers_all: List[Layer] = layers + [objective]

        # Back propagation function G=(g0 o g1 o ... o gn-1)
        self.__gradient = \
            compose(*[layer.gradient for layer in self._layers_all[::-1]])

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A unique name to identify a network"""
        return self._name

    @property
    def X(self) -> np.ndarray:
        """Batch input"""
        assert self._X, "Batch input not initialized or an invalid value."
        return self._X

    @property
    def N(self) -> int:
        """Batch size"""
        assert self._N > 0, "Batch size of X not initialized."
        return self._N

    @property
    def T(self) -> np.ndarray:
        """Labels"""
        assert self._T, "Label T not initialized or an invalid value."
        return self._T

    @property
    def L(self) -> Union[float, np.ndarray]:
        """Network objective value (Loss)"""
        assert self._L > 0, "Objective value L not initialized or an invalid value."
        return self._L

    @property
    def GN(self) -> List[Union[float, np.ndarray]]:
        """Numerical gradients of layers"""
        assert self._GN, "Numerical gradients GN of the network not initialized."
        return self._GN

    @property
    def dS(self) -> List[Union[float, np.ndarray]]:
        """Gradients dL/dS that have been used to update S in each layer"""
        assert self._dS, "Gradients dL/dS of the network not initialized."
        return self._dS

    @property
    def layers_inference(self) -> List[Layer]:
        """Inference layers"""
        assert self._layers_inference and len(self._layers_inference) > 0, \
            "Inference layers not initialized"
        return self._layers_inference

    @property
    def layers_objective(self) -> List[Layer]:
        """Objective layers"""
        assert self._layers_objective and len(self._layers_objective) > 0, \
            "Objective layers not initialized"
        return self._layers_objective

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

    @property
    def _inference(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network inference function I=fn-2 o ... o f0"""
        assert self.__inference, "Inference function I not initialized."
        return self.__inference

    @property
    def _objective(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network Objective function L=fn-1 o fn-2 o ... o f0"""
        assert self.__objective, "Objective function L not initialized."
        return self.__objective

    @property
    def _gradient(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network gradient function G=g0 o g1 o ... o gn-1"""
        assert self.__gradient, "gradient function G not initialized."
        return self.__gradient

    @property
    def logger(self) -> logging.Logger:
        """Instance logger"""
        assert self._logger, "logger not initialized"
        return self._logger

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def set(self, T: Union[float, np.ndarray]) -> NoReturn:
        """Setup a batch
        Args:
            T: labels
        """
        self._T = np.array(T) if isinstance(T, float) else T
        self._N = T.shape[0]
        for layer in self.layers_objective:
            layer.T = T

    def function(self, X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the network objective (Loss)
        Args:
            X: Batch input
        Returns:
            L: Objective value of the network (Loss)
        """
        X = np.array(X).reshape((1, -1)) if isinstance(X, float) else X
        self._X = np.array(X) if isinstance(X, float) else X
        assert self.X.shape[0] == self.N, \
            f"Batch size of X needs to be {self.N} but {self.X.shape}."
        self._L = self._objective(X)
        return self.L

    def gradient_numerical(self) -> List[Union[float, np.ndarray]]:
        """Numerical Gradients GN"""
        self._GN = [layer.gradient_numerical() for layer in self.layers_all]
        return self.GN

    def gradient(self) -> Union[float, np.ndarray]:
        """Back propagate gradients"""
        return self._gradient(np.array(1.0))

    def update(self) -> List[Union[float, np.ndarray]]:
        """Layer state S updates"""
        self._dS = [layer.update() for layer in self.layers_all]
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

    def train(self, X: Union[float, np.ndarray], T: Union[float, np.ndarray]):
        self.set(T)
        L = self.function(X)
        GN = self.gradient_numerical()
        dX = self.gradient()
        dS = self.update()
        D = self.validate()

        print(f"L is network loss is {L}")
        print(f"Gradient dL/dX is {dX}")
        print(f"Numerical gradients GN are {GN}")
        print(f"Analytical gradients dS are {dS}")
        print(f"Validations are {D}")
