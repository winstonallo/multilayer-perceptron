import numpy as np


class Neuron:
    def __init__(self, weights: list[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def output(self, inputs: list[float]) -> float:
        # Summation of weighted inputs with added bias.
        return np.dot(inputs, self.weights) + self.bias
