from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable

import numpy as np

from .tensor import Tensor


class Layer(ABC):
    """
    Layers are the building block of our neural net.
    Most nets will consist of several layers taking input and passing it to a subsequent layer.

    On a forward pass the layer applies a certain function to the inputs and saves what the inputs were.

    On a backward pass the layer propagates the gradients backward.
    """
    def __init__(self) -> None:
        self.inputs: Optional[Tensor] = None
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Apply layer's function to input tensor.
        :param inputs: A tensor of values.
        :return: A tensor with the resulting values of the layer's function.
        """
        pass

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """
        Update the layers gradients and propagate them backwards.
        :param grad: Next layer's gradients being passed in
        :return: New gradients after applying this layer's gradients
        """
        pass


class LinearLayer(Layer):
    """
    A linear layer takes inputs and applies a linear function to them:
    outputs = inputs @ weights + bias (Forward)
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super(LinearLayer, self).__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs  # Save inputs for calculating gradients on backward pass

        weights = self.params["w"]
        bias = self.params["b"]
        return inputs @ weights + bias

    def backward(self, grad: Tensor) -> Tensor:
        # TODO: Go over the math for this 👇
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


class ActivationLayer(Layer):
    """
    An activation layer just takes an input tensor and applies a function to it element-wise
    Useful link on different activations: http://cs231n.github.io/neural-networks-1/
    """
    def __init__(self, f: Callable[[Tensor], Tensor], f_prime: Callable[[Tensor], Tensor]):
        super(ActivationLayer, self).__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2


class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def relu(x: Tensor) -> Tensor:
    return np.maximum(0, x)


def relu_prime(x: Tensor) -> Tensor:
    return np.greater(x, 0).astype(int)


class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__(relu, relu_prime)

