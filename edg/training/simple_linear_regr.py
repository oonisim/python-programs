import os
import logging
import numpy as np
from simple_linear_regr_utils import generate_data, evaluate

TYPE_TENSOR = np.ndarray
TYPE_INT = np.int32
TYPE_FLOAT = np.float64
assert not isinstance(TYPE_FLOAT, (float, np.float))


def xavier(M: TYPE_INT, D: TYPE_INT) -> np.ndarray:
    """Xavier weight initialization for base-symmetric activations e.g. sigmoid/tanh
    Gaussian distribution with the standard deviation of sqrt(1/D) to initialize
    a weight W:(D,M) of shape (D, M), where D is the number of features to a node and
    M is the number of nodes in a layer.

    Args:
        M: Number of nodes in a layer
        D: Number of feature in a node to process
    Returns:
        W: weight matrix
    """
    assert all([D > 0, M > 0])
    return np.random.randn(M, D).astype(TYPE_FLOAT) / np.sqrt(D, dtype=TYPE_FLOAT)


def add_bias(X: TYPE_TENSOR) -> TYPE_TENSOR:
    """Add the bias term x0 instead of using a separate bias parameter b
    Args:
        X: Tensor of shape (N, D-1)
    Returns:
        X: Tensor of shape (N, D)
    """
    X = np.c_[
        np.ones(X.shape[0], dtype=TYPE_FLOAT),  # Add bias x0
        X.astype(TYPE_FLOAT)
    ]
    return X.astype(TYPE_FLOAT)


class SimpleLinearRegression:
    def __init__(self, iterations=15000, lr: TYPE_FLOAT = TYPE_FLOAT(0.1), log_level: int = logging.ERROR):
        """Instantiation
        Args:
            iterations: number of epoch to run
            lr: learning rate for the gradient descent
            log_level: logging level
        """
        logging.basicConfig(level=log_level)

        self.iterations = iterations    # number of iterations the fit method will be called
        self.lr: TYPE_FLOAT = lr        # The learning rate
        self.losses = []                # A list to hold the history of the calculated losses

        # --------------------------------------------------------------------------------
        # Consolidate (bias b, slope w) into one tensor W for single vectorized calculation.
        # NOTE: For small feature set, it is not memory efficient.
        # --------------------------------------------------------------------------------
        # self.W, self.b = None, None # the slope and the intercept of the model
        self.W = None

    def __loss(self, y: TYPE_TENSOR, y_hat: TYPE_TENSOR):
        """Objective function for the model performance
        Args:
            y: the actual output on the training set
            y_hat: the predicted output on the training set
        Returns:
            loss: the normalized sum of squared error
        """
        assert y_hat.shape == y.shape, \
            "Expected y.shape == y_hat.shape, got y.shape {} y_hat.shape {}".format(
                y.shape, y_hat.shape
            )

        N: TYPE_FLOAT = TYPE_FLOAT(len(y))
        loss = np.sum((y_hat - y) / N).astype(TYPE_FLOAT)
        self.losses.append(loss)

        return loss

    def __init_weights(self, X: TYPE_TENSOR):
        """Initialize the model parameters
        Args:
             X: The training set
        """
        # --------------------------------------------------------------------------------
        # Consolidate (bias b, slope w) into one tensor for single vectorized calculation
        # NOTE: For small feature set, it is not memory efficient.
        # --------------------------------------------------------------------------------
        # weights = np.random.normal(size=X.shape[1] + 1)
        # self.W = weights[:X.shape[1]].reshape(-1, X.shape[1])
        # self.b = weights[-1]

        D: TYPE_INT = TYPE_INT(X.shape[1])
        M: TYPE_INT = TYPE_INT(1)   # Number of hyper plane to use
        self.W = xavier(M, D)
        assert self.W.dtype == TYPE_FLOAT

    def __sgd(self, X: TYPE_TENSOR, y: TYPE_TENSOR, y_hat: TYPE_TENSOR) -> TYPE_TENSOR:
        """
        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        :return: Updated W
        """
        N: TYPE_INT = TYPE_INT(X.shape[0])
        D: TYPE_INT = TYPE_INT(X.shape[1])
        M: TYPE_INT = TYPE_INT(y_hat.shape[1])

        assert N == len(y), \
            "Expected number of rows in X and y must match, got X.rows[{}] y.rows [{}]".format(
                N, len(y)
            )
        assert y_hat.shape == y.shape, "Expected y.shape == y_hat.shape, got y.shape {}} y_hat.shape {}".format(
            y.shape, y_hat.shape
        )

        dW: TYPE_TENSOR = np.einsum("dn,nm->md", X.T, (y_hat - y)) / N
        assert dW.shape == (M, D)
        self.W: TYPE_TENSOR = self.W - self.lr * dW

        return self.W

    def fit(self, X, y):
        """
        :param X: The training set
        :param y: The true output of the training set
        :return:
        """
        self.__init_weights(X)

        # --------------------------------------------------------------------------------
        # Initial loss
        # --------------------------------------------------------------------------------
        y_hat = self.predict(X)
        loss = self.__loss(y, y_hat)
        print(f"Initial Loss: {loss}")

        # --------------------------------------------------------------------------------
        # Training loop
        # --------------------------------------------------------------------------------
        for i in range(self.iterations + 1):
            self.__sgd(X, y, y_hat)
            y_hat = self.predict(X)
            loss = self.__loss(y, y_hat)
            if not i % 100:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X: TYPE_TENSOR) -> TYPE_TENSOR:
        """Provide the prediction
        Args:
            X: Data to run the prediction
        Returns:
            y_hat: the prediction
        """
        assert isinstance(X, TYPE_TENSOR), "Expected {} type, got {}".format(
            TYPE_TENSOR, type(X)
        )
        # --------------------------------------------------------------------------------
        # Add bias if the input data is for (w, b) format
        # --------------------------------------------------------------------------------
        if X.shape[1] == (self.W.shape[1] - 1):
            X = add_bias(X)

        # --------------------------------------------------------------------------------
        # Geometory
        # N: Number of data points in X
        # D: Number of feature in each data in X
        # M: Number of regressors to use.
        # --------------------------------------------------------------------------------
        N: TYPE_INT = TYPE_INT(X.shape[0])
        D: TYPE_INT = TYPE_INT(X.shape[1])
        M: TYPE_INT = TYPE_INT(self.W.shape[0])
        assert D == self.W.shape[1], \
            "Expected D == X.shape[1] == W.shape[1] got X.shape {} W.shape {}".format(
                X.shape, self.W.shape
        )

        # --------------------------------------------------------------------------------
        # Inner product to measure the proximity between X and W
        # --------------------------------------------------------------------------------
        y_hat = np.einsum("nd,dm->nm", X, self.W.T).astype(TYPE_FLOAT)
        assert y_hat.shape == (N, M)

        return y_hat

    def save(self, path_to_file: str):
        """Save model to a file
        Args:
            path_to_file: path to the file
        Raises: RuntimeError for I/O errors.
        """
        assert self.W is not None and isinstance(self.W, np.ndarray)
        try:
            with open(path_to_file, 'wb') as f:
                np.save(f, self.W)
        except OSError as e:
            logging.error("save(): failed to save to {} due to {}".format(path_to_file, e))
            raise RuntimeError from e

    def load(self, path_to_file: str):
        """Load model from a file
        Args:
            path_to_file: path to the file
        Raises: RuntimeError for I/O errors.
        """
        try:
            with open(path_to_file, 'rb') as f:
                self.W = np.load(f)
        except OSError as e:
            logging.error("load(): failed to load from {} due to {}".format(path_to_file, e))
            raise RuntimeError from e


def main():
    X_train, y_train, X_test, y_test = generate_data()
    X_test = X_test.astype(TYPE_FLOAT)
    y_test = y_test.astype(TYPE_FLOAT)

    # --------------------------------------------------------------------------------
    # Add bias x0 to the training date to consolidate the model parameters (w, b) to W.
    # Leave the test data as-is. predict() handle the bias x0.
    # --------------------------------------------------------------------------------
    X_train = add_bias(X_train)
    y_train = y_train.astype(TYPE_FLOAT)

    # --------------------------------------------------------------------------------
    # Train the model
    # --------------------------------------------------------------------------------
    model = SimpleLinearRegression()
    model.fit(X_train,y_train)

    # --------------------------------------------------------------------------------
    # Evaluate the model
    # --------------------------------------------------------------------------------
    predicted = model.predict(X_test)
    if evaluate(model, X_test, y_test, predicted):
        # --------------------------------------------------------------------------------
        # Save the model
        # --------------------------------------------------------------------------------
        model.save("../model/model.npy")
    else:
        logging.error("Evaluation failed")


if __name__ == "__main__":
    main()
