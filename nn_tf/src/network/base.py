"""Base for neural network implementations
Network is a higher order composite of (inference, objective) each of which
can be a network itself. Hence network class is subclass of Layer.

objective:
    It is a replaceable trainer who e-valuates the performance of "inference".
    The objective setter of the Layer class sets an objective layer of the
    network. It would be a MSE layer or a cross entropy log loss layer, but
    also it can be a network.

inference:
    It can be either
    1. a stateful terminal layer (e.g. Matmul)
    2. a stateless terminal layer (e.g. ReLU)
    3. a composite (non-network) layer (e.g Sequential)
    4. a network, creating a higher-order recursive composite

    A stateful layer can learn to estimate q0(X) via the e-valuation from
    the objective and generates a belief on the population q1(X).

    A network has interfaces, besides those of Layer, which a terminal and
    a composite layer does not have.

    - train
    - L/Loss
    - history


"function" of a "network".
    Have a clear separation between
    1. function at training on the sample q0(X)
    2. function at inference (including test) on the population q1(x).

    "Function at training" generates an estimate or approximation of the
    population q1(x) based on the sample q0(X). The objective function
    evaluates the estimation to improve the estimate by the model on q0(X).
    We treat q0(X) as "X" during the training.

    * network.function(q0(X)) is Estimation of q0(X)
    * network.objective(q0(X)) is Evaluation of the estimation on q0(X)

    "Function at inference" generates a belief on the general population q1(X)
    and we treat q1(X) as "X".

    * network.prediction(q1(X)) is Belief.

    To avoid confusions, not use "**function** at inference" but simply use
    "prediction" to represent it.

Interfaces:
    train(q0(X), T):
        Estimate AND evaluate..

    predict(q1(X)):
        Infer the population.

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
from layer.schemes import (
    FUNCTION_LAYERS,
    OBJECTIVE_LAYERS,
)
from layer.schemes_composite import (
    COMPOSITE_LAYERS
)


class Network(Layer):
    """Neural network base class"""

    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def T(self) -> np.ndarray:
        """Label in index format"""
        return super().T

    @T.setter
    def T(self, T: Union[np.ndarray, int]):
        """Label setter
        The inference layer can be a network which needs T. Hence set on both
        inference and objective layers.
        """
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
        assert isinstance(self._history, list) and len(self._history) > 0, \
            "Objective value L not initialized or an invalid value."
        return self._history

    @property
    def GN(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Numerical gradients of layers"""
        assert isinstance(self._GN, list) and len(self._GN) > 0, \
            "Numerical gradients GN of the network not initialized."
        return self._GN

    @property
    def layer_inference(self) -> Layer:
        """Inference layers"""
        assert isinstance(self._layer_inference, FUNCTION_LAYERS + COMPOSITE_LAYERS), \
            "Inference layer not initialized"
        return self._layer_inference

    @property
    def layer_objective(self) -> Layer:
        """Objective layers"""
        assert isinstance(self._layer_objective, OBJECTIVE_LAYERS + COMPOSITE_LAYERS), \
            "Objective layers not initialized"
        return self._layer_objective

    @property
    def layers_all(self) -> List[Layer]:
        """(inference, objective) in the network.
        Keep the structure simple (inf, obj) only. Use currying to compose a complex.
        """
        assert self._layers_all and len(self._layers_all) == 2, \
            "Network need to have 1 inference layer and 1 objective layer."
        return self._layers_all

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
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

        assert isinstance(name, str) and len(name) > 0

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])

        # --------------------------------------------------------------------------------
        # Objective (Loss)
        # --------------------------------------------------------------------------------
        self._L: Union[TYPE_FLOAT, np.ndarray] = []
        self._history: List[Union[TYPE_FLOAT, np.ndarray]] = []

        self._GN: List[Union[TYPE_FLOAT, np.ndarray]] = []   # Numerical gradients GN of layers

        # --------------------------------------------------------------------------------
        # Inference layers in the network
        # --------------------------------------------------------------------------------
        self._layer_inference: Layer = None
        self._layer_objective: Layer = None
        self._function: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
        self._predict: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
        self._gradient: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
        self._update: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None

        self._layers_all: List[Layer] = []

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
        self.X = X
        self._Y = self._function(X)
        return self.Y

    def gradient_numerical(
            self, h: TYPE_FLOAT = 1e-5
    ) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Numerical Gradients GN"""
        self._GN = [__layer.gradient_numerical() for __layer in self.layers_all]
        return self.GN

    def gradient(self, dY: Union[np.ndarray, TYPE_FLOAT]) -> Union[TYPE_FLOAT, np.ndarray]:
        """Back propagate gradients"""
        dY = np.array(dY).astype(TYPE_FLOAT) if isinstance(dY, TYPE_FLOAT) else dY
        assert isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT
        return self._gradient(dY)

    def update(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """
        Responsibility:
            Update network state S via the gradient descent on the inference layer.

            The objective may have a state (e.g. change e-valuation method/value)
            based on the performance of the inference. However, for now, the state
            is to do the prediction, which is done by the inference layer, not the
            objective.

            T. T is stateful but it is constant during a cycle, hence regarded as
            state-less during the cycle in which update() is called.

        Returns:
            dS: List of the gradients on layer state(s)
        """
        self._dS = self.layer_inference.update()
        return self.dS

    def train(
            self,
            X: Union[TYPE_FLOAT, np.ndarray],
            T: Union[TYPE_FLOAT, np.ndarray],
            run_validations: bool = False
    ):
        """
        Args:
            X: batch training data in shape (N,M)
            T: label of shape (N,) in the index format
            run_validations: Flat if run the validations e.g. numerical gradient check
        Returns:
            Model S: Updated state of the network
        """
        self.X = X.astype(TYPE_FLOAT)
        self.T = T.astype(TYPE_LABEL)

        # --------------------------------------------------------------------------------
        # Forward path
        # --------------------------------------------------------------------------------
        self._Y = self.function(self.X).astype(TYPE_FLOAT)
        self._L = self.objective(self.Y).astype(TYPE_FLOAT)
        self._history.append(self.L)

        # --------------------------------------------------------------------------------
        # Backward path
        # --------------------------------------------------------------------------------
        self._dX = self.gradient(TYPE_FLOAT(1)).astype(TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Gradient descent
        # --------------------------------------------------------------------------------
        self.update()

        # --------------------------------------------------------------------------------
        # Info
        # --------------------------------------------------------------------------------
        self.logger.info("Network[%s]: Loss is %s", self.name, self.L)
        self.logger.info("Gradient dL/dX is %s", self.dX)
        self.logger.info("Analytical gradients dS are %s\n", self.dS)

        return self.S

    def predict(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """Calculate the prediction on X
        TODO: Research if put through softmax, etc. The same used by the objective.
        Since the gradient descent process includes dL/dA at the softmax etc at the
        objective layer, the state should be depending on what the softmax has done.
        Hence it seems rational to incorporate the same activation at the objective.
        """
        assert isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT, \
            f"Only np array of type {TYPE_FLOAT} is accepted"

        return self._predict(X).astype(TYPE_LABEL)
