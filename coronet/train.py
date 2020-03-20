"""
Simple function to train net.
"""

from .nn import NeuralNet
from .loss import Loss, MSE
from .data import DataIterator, BatchIterator
from .optimizers import Optimizer, StochasticGradientDescent
from .tensor import Tensor


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int,
          loss: Loss = MSE(),
          iterator: DataIterator = BatchIterator(),
          optimizer: Optimizer = StochasticGradientDescent()) -> None:
    for epoch in range(0, num_epochs):
        epoch_loss: float = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)  # Get predictions
            grad = loss.grad(predicted, batch.targets)  # Calculate gradients using loss function
            epoch_loss += loss.loss(predicted, batch.targets)  # Accumulate to epoch loss
            net.backward(grad)  # Backpropagate through net and update gradients
            optimizer.step(net)  # Update params using updated gradients
        print(epoch, epoch_loss)
