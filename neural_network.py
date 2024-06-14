import numpy as np


class Neuron:
    # The functionality of this class is abstracted away in NeuronLayer
    # using the dot product of the weights and the inputs.

    def output(self, weights: list[float], inputs: list[float], bias: float) -> float:
        # Summation of weighted inputs with added bias.
        return np.dot(inputs, weights) + bias


class NeuronLayer:

    def run(weights: list[list[float]], inputs: list[float], biases: list[float]) -> list[float]:
        # Get the outputs of the entire layer in one line using the dot product.
        # In this case, np.dot() multiplies the weights matrix (X x Y)
        # with the inputs vector (1 x X).
        # See the Neuron class.
        return np.dot(weights, inputs) + biases
    
    def batch(weights: list[list[float]], inputs: list[list[float]], biases: list[float]) -> list[list[float]]:
        # This gets the layer outputs for a whole batch of inputs at a time.
        # Inputs and weights are both X x Y, therefore we must transpose the weights
        # in order to convert them into Y x X and make the dot product possible.
        return np.dot(inputs, np.array(weights).T) + biases
