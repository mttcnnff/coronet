import numpy as np

from coronet.train import train
from coronet.nn import NeuralNet
from coronet.layers import LinearLayer, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    LinearLayer(input_size=2, output_size=2),
    Tanh(),
    LinearLayer(input_size=2, output_size=2)
])


def xor():
    train(net, inputs, targets, 5000)

    for x, y in zip(inputs, targets):
        predicted = net.forward(x)
        print(x, predicted, y)


if __name__  == "__main__":
    xor()

