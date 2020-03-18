from abc import ABC, abstractmethod

from .nn import NeuralNet


class Optimizer(ABC):
    @abstractmethod
    def step(self, net: NeuralNet) -> None:
        pass


class StochasticGradientDescent(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
