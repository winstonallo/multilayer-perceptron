import numpy as np
from numpy import ndarray


class DenseLayer:
    # This class simplifies the creation of neuron layers by
    # randomly assigning weights on a Gaussian distribution,
    # and setting bias to 0.

    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: ndarray) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLUActivation:
    # Implementation of the ReLu activation function, which captures
    # non-linearity by setting negative values to 0.
    #
    # Formula: y = max(0, x)

    def forward(self, inputs: ndarray) -> None:
        self.output = np.maximum(0, inputs)


class SoftmaxActivation:
    # Implementation of the Softmax activation function, which is used
    # in the output layer of a neural network to produce probabilities
    # by ensuring each output is in a (0, 1) range and the sum
    # of all outputs equals to 0.
    #
    # Formula: e^x / sum(e^x)

    def forward(self, inputs: ndarray) -> None:
        # For each row, we subtract its highest value from all others before
        # taking the exponents. This prevents the exponents from getting
        # too big.
        # We use keepdims=True in order to keep the dimensions of the original array.
        # This is VERY important, as mismatched dimensions lead to unrelated values
        # being subtracted from the rows.
        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # After getting the exponent for each normalized value, we divide each of them
        # by the sum of all others in their respective row.
        #
        # Example, for this array: [x y z]
        # x = e^x / e^x + e^y + e^z
        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
        self.output = probabilities
