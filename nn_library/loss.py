"""
A loss function measures how good predictions are.
Can be used to adjust parameters of our network
"""


import numpy as np
from nn_library.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    MSE is mean squared error, though just going tod o total squared error
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)