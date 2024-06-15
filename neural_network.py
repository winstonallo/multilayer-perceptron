import numpy as np
from numpy import ndarray


class Neuron:
    # The functionality of this class is abstracted away in NeuronLayer
    # using the dot product of the weights and the inputs.

    def output(self, weights: list[float], inputs: list[float], bias: float) -> float:
        # Summation of weighted inputs with added bias.
        return np.dot(inputs, weights) + bias


class NeuronLayer:

    def __init__(self, weights: ndarray, inputs: ndarray, biases: ndarray):
        self.weights = np.array(weights)
        self.inputs = np.array(inputs)
        self.biases = np.array(biases)

    def run(self, weights: ndarray, inputs: ndarray, biases: ndarray) -> ndarray:
        # Get the outputs of the entire layer in one line using the dot product.
        # In this case, np.dot() multiplies the weights matrix (X x Y)
        # with the inputs vector (1 x X).
        # See the Neuron class.
        return np.dot(weights, inputs) + biases

    def output(self) -> ndarray:
        # This gets the layer outputs for a whole batch of inputs at a time.
        # Inputs and weights are both X x Y, therefore we must transpose the weights
        # in order to convert them into Y x X and make the dot product possible.
        return np.dot(self.inputs, self.weights.T) + self.biases


class DenseLayer:
    # This class simplifies the creation of neuron layers by
    # randomly assigning weights on a Gaussian distribution,
    # and setting bias to 0.

    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: ndarray) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLuActivation:
    # Implementation of the ReLu activation function, which captures
    # non-linearity by setting negative values to 0.
    #
    # Formula: y = max(0, x)

    def forward(self, inputs: ndarray) -> None:
        self.output = np.maximum(0, inputs)


class SoftMaxActivation:
    # Implementation of the SoftMax activation function, which is used
    # in the output layer of a neural network to produce probabilities
    # by ensuring each output is in a (0, 1) range and the sum
    # of all outputs equals to 0.
    #
    # Formula: e^x / sum(e^x)

    def forward(self, inputs: ndarray) -> None:
        # We take the maximum value by row (axis=1) and subtract
        # it from the exponential values in order to prevent them
        # from becoming too big.
        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
        self.output = probabilities
