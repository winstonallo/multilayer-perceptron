import numpy as np
from numpy import ndarray


# The following classes are used to create layers in a neural network.
# Each class has a forward method that calculates the output of the layer
# based on the input it receives.
#
# Notation:
#   - x: inputs (array)
#   - y: output (array)
#   - W: weights (array)
#   - b: biases (array)
#   - n_{x | neurons}: number of {x | neurons}


class DenseLayer:

    def __init__(self, n_x: int, n_neurons: int):
        self.W = 0.01 * np.random.randn(n_x, n_neurons)
        self.b = np.zeros((1, n_neurons))

    def forward(self, x: ndarray) -> None:
        self.y = np.dot(x, self.W) + self.b


class SigmoidActivation:
    # Implementation of the sigmoid activation function, which
    # squashes values between 0 and 1.
    #
    # Formula: y = 1 / (1 + e^-x)

    def forward(self, x: ndarray) -> None:
        self.y = 1 / (1 + np.exp(-x))


class ReLUActivation:
    # Implementation of the ReLu activation function, which captures
    # non-linearity by setting negative values to 0.
    #
    # Formula: y = max(0, x)

    def forward(self, x: ndarray) -> None:
        self.y = np.maximum(0, x)


class SoftmaxActivation:
    # Implementation of the Softmax activation function, which is used
    # in the output layer of a neural network to produce probabilities
    # by ensuring each output is in a (0, 1) range and the sum
    # of all outputs equals to 0.
    #
    # Formula: e^x / sum(e^x)

    def forward(self, x: ndarray) -> None:
        # For each row, we subtract its highest value from all others before
        # taking the exponents. This prevents the exponents from getting
        # too big.
        # We use keepdims=True in order to keep the dimensions of the original array.
        # This is VERY important, as mismatched dimensions lead to unrelated values
        # being subtracted from the rows.
        exponential_values = np.exp(x - np.max(x, axis=1, keepdims=True))

        # After getting the exponent for each normalized value, we divide each of them
        # by the sum of all others in their respective row.
        #
        # Example, for this array: [x y z]
        # x = e^x / e^x + e^y + e^z
        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
        self.y = probabilities
