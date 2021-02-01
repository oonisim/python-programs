"""Gradient descent base"""


class Optimizer:
    """Gradient descent optimization base class implementation"""
    @property
    def lr(self) -> float:
        """Learning rate of the gradient descent"""
        return self._lr

    def __init__(self, lr=0.01):
        self._lr = lr

    def update(self, W, dW):
        """Default method to update the weight matrix W
        Args:
            W: weight matrix to update
            dW: gradient of dL/dW, the impact of dW on the system output L
        Returns:
            Updated W
        """
        assert False, "You need to override and implement the update method"



