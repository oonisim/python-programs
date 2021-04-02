"""Objective function layer implementations
"""
import logging
from functools import partial
from typing import (
    List,
    Dict,
    Union,
    Callable,
    NoReturn
)
import numpy as np
import common
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL,
)
from layer.constants import (
    LOG_LOSS_GRADIENT_ACCEPTANCE_VALUE
)
from common.function import (
    transform_X_T,
    sigmoid,
    softmax,
    generic_cross_entropy_log_loss,
    sigmoid_cross_entropy_log_loss,
)
from layer.base import Layer
from layer.constants import (
    _NAME,
    _NUM_NODES,
    _LOSS_FUNCTION,
    _LOG_LEVEL,
    _SCHEME,
    _PARAMETERS
)


class CrossEntropyLogLoss(Layer):
    """Cross entropy log loss class
    Combined with the log loss because calculating gradients separately is not simple.
    When combined, the dL/dX, impact on L by input delta dX, is (P - T)/N.

    Note:
        Number of nodes M == number of feature D at the objective/loss layer.
        T has either in the OHE (One Hot Encoder) label format or index label format.
        For OHE T.shape[0] == N and T.shape[1] == M == D. For index, T.size == N.

    TODO:
        Add regularization cost 0.5 * l2 * np.sum(W ** 2) to L. Need a way to access
        W in each matmul layer.
    """
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return CrossEntropyLogLoss.specification(name="loss001", num_nodes=3)

    @staticmethod
    def specification(
            name: str,
            num_nodes: int,
            loss_function: str = common.softmax_cross_entropy_log_loss.__qualname__
    ):
        """Generate ReLU specification
        Args:
            name: layer name
            num_nodes: number of nodes (outputs) in the layer
            loss_function: cross entropy log loss function
        """
        assert loss_function in common.LOSS_FUNCTIONS, \
            "Invalid loss function %s. Must be one of %s" \
            % (loss_function, list(common.LOSS_FUNCTIONS.keys()))

        return {
            _SCHEME: CrossEntropyLogLoss.__qualname__,
            _PARAMETERS: {
                _NAME: name,
                _NUM_NODES: num_nodes,
                _LOSS_FUNCTION: loss_function
            }
        }

    @staticmethod
    def build(parameters: Dict):
        assert (
            _NAME in parameters and
            _NUM_NODES in parameters and
            _LOSS_FUNCTION in parameters and
            parameters[_LOSS_FUNCTION] in common.LOSS_FUNCTIONS
        )

        return CrossEntropyLogLoss(
            name=parameters[_NAME],
            num_nodes=parameters[_NUM_NODES],
            log_loss_function=common.LOSS_FUNCTIONS[parameters[_LOSS_FUNCTION]],
            log_level=parameters[_LOG_LEVEL] if _LOG_LEVEL in parameters else logging.ERROR
        )

    # ================================================================================
    # Instance
    # ================================================================================
    def __init__(
            self, name:
            str, num_nodes: int,
            log_loss_function: Callable = generic_cross_entropy_log_loss,
            log_level: int = logging.ERROR
    ):
        """Initialize the layer
        Args
            name: Instance ID
            num_nodes: Number of nodes in the layer
            f: function f(X, T) to calculate the cross entropy log loss 
            log_level: Logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        def _objective(X: np.ndarray) -> Union[float, np.ndarray]:
            """Output layer objective function
            As this is the last layer, the layer function itself is the objective function.
            Hence the objective function of the output layer simply pass through X
            """
            assert X.ndim == 0, "The output of the log loss should be of shape ()"
            return X

        self.objective = _objective     # objective function f(X, T)

        if log_loss_function == sigmoid_cross_entropy_log_loss:
            assert num_nodes == 1   # binary classification
        else:
            assert num_nodes > 1    # multi label categorical classification.

        self._log_loss_function = log_loss_function

        # At the objective layer, features in X is the same with num nodes.
        self._D = num_nodes
        self._P: np.ndarray = np.empty(())  # Probabilities of shape (N, M)
        self._J: np.ndarray = np.empty(())  # Cross entropy log loss of shape (N,).
        # Use Y for output in the consistent manner.
        # self._L: np.ndarray = -np.inf       # Objective value of shape ()

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def P(self) -> np.ndarray:
        """Activation outputs of shape (N,M). Probabilities when activation=softmax"""
        assert (
                isinstance(self._P, np.ndarray) and self._P.size > 0
                and self._P.shape == (self.N, self.M)
        ), "P is not initialized"
        return self._P

    @property
    def J(self) -> np.ndarray:
        """Cross entropy log loss of shape(N,)"""
        assert isinstance(self._J, np.ndarray) and self._J.size == self.N, \
            "J is not initialized"
        return self._J

    @property
    def L(self) -> np.ndarray:
        """Cross entropy log loss of shape()
        Alias of the layer output Y
        """
        assert isinstance(self._Y, np.ndarray) and self._Y.size == 1, \
            "Y is not initialized"
        return self._Y

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[np.ndarray, float]) -> np.ndarray:
        """Log Loss layer output L
        Pre-requisite:
            T has been set before calling.
        Note:
            The Log Loss output is normalized by the batch size N.
            Otherwise the NN loss is dependent on the batch size.

            For instance, with a MNIST image, the loss or NN performance should be
            per-image, so that the loss can be a universal unit to compare.
            It should NOT be dependent on what batch size was used.

            L = cross_entropy_log_loss(activation(X), T, log_loss_function)) / N
            or
            L =  softmax_cross_entropy_log_loss(X, T) / N for softmax
            or
            L =  sigmoid_cross_entropy_log_loss(X, T) / N for sigmoid

        Args:
            X: Input of shape (N,M) to calculate the probabilities for the M nodes.
        Returns:
            L: Objective value of shape ()
        """
        name = "function"
        assert self.T.size > 0 and self.M == self.D, \
            "Pre-requisite: T has been set before calling."

        # --------------------------------------------------------------------------------
        # Validate X, T and transform them to be able to use numpy tuple-like indexing.
        # P[
        #   (0,3),
        #   (1,5)
        # ]
        # --------------------------------------------------------------------------------
        self.X, self.T = transform_X_T(X, self.T)
        self.logger.debug(
            "layer[%s].%s: After transform_X_T, X.shape %s T.shape %s",
            self.name, name, self.X.shape, self.T.shape
        )

        assert (
                # Index labels P:(N,M), T:(N,) e.g. T[0,4,2] P[[1.,0.,...],[...],[...]] or
                # Binary oHE labels P(N,M), T(N,M) e.g T[[0],[1],[0]], P[[0,1],[0.9],[0.]]
                self.X.ndim >= 2 and self.T.ndim in {1, 2} and
                self.X.shape[0] == self.T.shape[0] and
                self.X.shape[1] == self.M       # M=1 for logistic binary
            ), \
            "X shape %s with T.shape %s does not match with the Layer node number M[%s]" \
            % (self.X.shape, self.T.shape, self.M)

        # --------------------------------------------------------------------------------
        # Activations P:(N, M) for each label m in each batch n.
        # Cross entropy log loss J:(N,) where j(n) for each batch n (n: 0, ..., N-1).
        # Gradient dJ/dP -> -T/P
        # --------------------------------------------------------------------------------
        self._J, self._P = self._log_loss_function(X=self.X, T=self.T)

        # --------------------------------------------------------------------------------
        # Total batch loss _L.
        # d_L/dJ = 1 at this point because dsum(J)/dJ = 1
        # --------------------------------------------------------------------------------
        _L = np.sum(self.J, axis=-1)

        # --------------------------------------------------------------------------------
        # Normalize L with the batch size N to be independent from the batch size.
        # dL/dJ = 1/N because f(X)=X/N -> df(X)/dX = 1/N
        # Convert scalar back to np.ndarray as np.sum() gives scalar.
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # --------------------------------------------------------------------------------
        L = np.array(_L / self.N, dtype=TYPE_FLOAT)
        self._Y = L         # L is alias of Y.

        self.logger.debug("Layer[%s].%s: L = %s", self.name, name, self.Y)
        return self.Y

    def gradient(self, dY: Union[np.ndarray, float] = float(1)) -> Union[np.ndarray, float]:
        """Calculate the gradient dL/dX, the impact on F by the input dX.
        L: Output of the layer. Alias is Y.
        J = cross_entropy_log_loss(P,T)
        dY: dL/dL=1, impact on L by the layer output Y=L.
        dJ: dL/dJ = 1/N is the impact on Y by dJ where .
        dP: dJ/dP = -T/P is the impact on J by the activation output dP.
        dP/dX: impact on P by the activation input dX
        dL/dX = dL/dJ * (dJ/dP * dP/dX) = 1/N * (P - T)

        TODO:
            Clarify what to do with the gradient for elements where T=0.
            For T=1, gradient (_P-T) is calculated, but no calculation for T=0.
            Need to make sure this is a correct operation.

            If gradient descent is not required for T=0 elements, might need
            to zero-clear the gradients for those elements.

        Note:
            For both softmax and sigmoid, dL/dX is (P-T)/N, so designed.
            P=sigmoid(X) or softmax(X). 0 <= P <=1 are the limits, T is 0 or 1.
            Hence, -1 <= dL/dX <= 1

        Args:
            dY: Gradient, impact by the loss dY, given from the post layer.
        Returns:
            dL/dX: (P-T)/N of shape (N, M)
        """
        name = f"Layer[{self.name}].gradient()"
        dY = np.array(dY) if isinstance(dY, float) else dY

        # --------------------------------------------------------------------------------
        # Shapes of dL/dY and Y are the same because L is scalar.
        # --------------------------------------------------------------------------------
        assert \
            (isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT) and \
            (dY.shape == self.Y.shape), \
            "dY/dY shape needs %s of type float but %s of type %s" % \
            (self.Y.shape, dY.shape, dY.dtype)

        # dL/dJ is 1/N of shape (N,) but transform into shape (N,1) to np-broadcast.
        dJ: np.ndarray = (dY * np.ones(self.N, dtype=TYPE_FLOAT) / float(self.N)).reshape(-1, 1)

        # --------------------------------------------------------------------------------
        # Calculate the layer gradient
        # --------------------------------------------------------------------------------
        if self.T.ndim == self.P.ndim:
            # --------------------------------------------------------------------------------
            # T is OHE if T.ndim==2 else index with T.ndim==1
            # --------------------------------------------------------------------------------
            self.logger.debug("Layer[%s].%s: Label is in OHE format", self.name, name)
            assert (self.T.size == self.P.size) and (self.T.shape == self.P.shape), \
                "T.shape %s and P.shape %s should be the same for the OHE labels." \
                % (self.T.shape, self.P.shape)

            # --------------------------------------------------------------------------------
            # Is this correct?
            # For T in index label, pick up P elements p for which t is 1, hence not using
            # p for which t is 0, ignoring the impact from t=0 elements.
            #
            # [Q] Probably it is correct for Softmax only?
            # [A] It is correct here because for sigmoid/binary classification, T is
            # always OHE and P.shape[1]==T.shape[1]==1. It cannot be index label.
            # e.g. For T[[0],[1],[0]], P[[0,2],[0,9],[0.0]], if T is index label, P.shape[1]
            # needs to be 2 to be able to use t=1 as index such as P[::, 1], which causes
            # index out of bound.
            # --------------------------------------------------------------------------------
            dX = dJ * (self.P - self.T)     # (N,M) * (N,M)

        else:
            self.logger.debug("%s: Label is index format", name)
            # --------------------------------------------------------------------------------
            # np.copy() is a shallow copy and will not copy object elements within arrays.
            # To ensure all elements within an object array are copied, use copy.deepcopy().
            # Here, reusing dP elements hence shallow copy.
            # --------------------------------------------------------------------------------
            dX = self.P.copy()

            # --------------------------------------------------------------------------------
            # Calculate dJ/dX=(P-T) using numpy tuple indexing. The tuple size must be the same.
            # e.g. select P[n=0][m=2] and P[n=3][m=4]:
            # P[
            #   (0, 3),
            #   (2, 4)
            # ]
            # --------------------------------------------------------------------------------
            rows = np.arange(self.N)    # tuple index for rows
            cols = self.T               # tuple index for columns
            assert rows.shape == cols.shape, \
                f"numpy tuple indices {rows.shape} and {cols.shape} need to be the same."

            # Extract T=1 elements of P and calculate (P-T) = (P-1)
            # dX shape is (N,M), not (N,)
            dX[
                rows,
                cols
            ] -= 1.0
            # dF/dY * dY/dJ * (P-T) = dF/dY * (P-T) / N
            np.multiply(dJ, dX, out=dX)

        assert np.all(np.abs(dX) <= 1), \
            f"Need the gradient dL/dX between [-1, 1] but\n{dX}"

        self.logger.debug("%s: dL/dX is \n%s.\n", name, dX)
        return dX

    def gradient_numerical(
            self, h: float = 1e-5
    ) -> List[Union[float, np.ndarray]]:
        GN: Union[float, np.ndarray] = super().gradient_numerical()[0]

        # Analytical gradient dL/dX is (P-T)/N whose range is [-1,1].
        # Numerical gradient GN should not be far away from the boundary.
        assert np.all(np.abs(GN) < LOG_LOSS_GRADIENT_ACCEPTANCE_VALUE), \
            "%s: numerical dL/dX needs between (-1.2, 1.2) but \n%s\n"\
            % (f"Layer[{self.name}].gradient_numerical()", GN)

        return [GN]

    def predict(self, X):
        """
        Responsibility:
            Generate a concrete prediction 0/1 for binary classification
            and an index for categorical classification

            For binary classification where M=1, 1/True if Xi > 0 else 0.

            For categorical classification, argmax(X, axis=1) that identifies
            the class that gives the max probability.
        Args:
            X: scores
        Returns:
            Predictions
        """
        assert isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT, \
            f"Only np array of type {TYPE_FLOAT} is accepted"
        if X.ndim <= 1:
            X = np.array(X).reshape(1, -1)

        if self._log_loss_function == sigmoid_cross_entropy_log_loss:
            return (X > 0).astype(TYPE_LABEL)
        else:
            return np.argmax(softmax(X), axis=1).astype(TYPE_LABEL)
