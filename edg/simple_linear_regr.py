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
        X
    ]
    return X


class SimpleLinearRegression:
    def __init__(self, iterations=15000, lr: TYPE_FLOAT = 0.1):
        self.iterations = iterations # number of iterations the fit method will be called
        self.lr: TYPE_FLOAT = lr # The learning rate
        self.losses = [] # A list to hold the history of the calculated losses
        # self.W, self.b = None, None # the slope and the intercept of the model
        self.W = None

    def __loss(self, y: TYPE_TENSOR, y_hat: TYPE_TENSOR):
        """

        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error

        """
        assert y_hat.shape == y.shape, \
            "Expected y.shape == y_hat.shape, got y.shape {} y_hat.shape {}".format(
                y.shape, y_hat.shape
            )

        N = len(y)
        loss = np.sum((y_hat - y) / N)

        self.losses.append(loss)
        return loss

    def __init_weights(self, X: TYPE_TENSOR):
        """
        :param X: The training set
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

    def __sgd(self, X, y, y_hat) -> TYPE_TENSOR:
        """
        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        :return: Updated W
        """
        N = X.shape[0]
        D = X.shape[1]
        M = y_hat.shape[1]

        assert N == len(y), \
            "Expected number of rows in X and y must match, got X.rows[{}] y.rows [{}]".format(
                N, len(y)
            )
        assert y_hat.shape == y.shape, "Expected y.shape == y_hat.shape, got y.shape {}} y_hat.shape {}".format(
            y.shape, y_hat.shape
        )

        dW: TYPE_TENSOR = np.einsum("dn,nm->md", X.T, (y_hat - y)) / N
        assert dW.shape == (M, D)
        self.W = self.W - self.lr * dW

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
        # Training
        # --------------------------------------------------------------------------------
        for i in range(self.iterations + 1):
            self.__sgd(X, y, y_hat)
            y_hat = self.predict(X)
            loss = self.__loss(y, y_hat)
            if not i % 100:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X: TYPE_TENSOR) -> TYPE_TENSOR:
        """
        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        N = X.shape[0]
        D = X.shape[1]
        M = self.W.shape[0]
        assert D == self.W.shape[1], \
            "Expected D == X.shape[1] == W.shape[1] got X.shape {} W.shape {}".format(
                X.shape, self.W.shape
        )
        y_hat = np.einsum("nd,dm->nm", X, self.W.T)

        assert y_hat.shape == (N, M)
        return y_hat


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_data()

    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    model = SimpleLinearRegression()
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    evaluate(model, X_test, y_test, predicted)
