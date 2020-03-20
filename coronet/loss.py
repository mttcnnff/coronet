"""
Loss functions will inform us on how our predictions are vs expected results.

We can use them to adjust our parameters as we train.
"""
from abc import ABC, abstractmethod

import numpy as np

from .tensor import Tensor


class Loss(ABC):
    @abstractmethod
    def loss(self, predicted: Tensor, expected: Tensor) -> float:
        pass

    @abstractmethod
    def grad(self, predicated: Tensor, expected: Tensor) -> Tensor:
        pass


class MSE(Loss):
    """
    MSE (Mean Squared Error) is the average squared difference
    between the predicted and expected output values
    """

    def loss(self, predicted: Tensor, expected: Tensor) -> float:
        return np.sum((predicted - expected) ** 2) / len(predicted)

    def grad(self, predicted: Tensor, expected: Tensor) -> Tensor:
        return (2 / len(predicted)) * np.sum(predicted - expected)
