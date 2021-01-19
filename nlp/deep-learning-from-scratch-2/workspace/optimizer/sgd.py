"""Gradient descent algorithm implementations"""
import numpy as np


class Optimizer:
    @property
    def lr(self) -> float:
        """Learning rate of the gradient descent"""
        return self._lr

    def __init__(self, lr=0.01):
        self._lr = lr

    def update(self, W, dW):
        return np.subtract(W, self.lr * dW, out=W)
        return W


class SGD(Optimizer):
    """Stochastic gradient descent """
    def update(self, W, dW):
        np.subtract(W, self.lr * dW, out=W)
        return W
