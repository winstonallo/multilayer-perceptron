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
#   - L: loss
#   - d{1}_d{2}: partial derivative of {1} with respect to {2}


class DenseLayer:

    def __init__(self, n_x: int, n_neurons: int, learning_rate: float):
        self.W = 0.01 * np.random.randn(n_x, n_neurons) / np.sqrt(n_x)
        self.b = np.zeros((1, n_neurons))
        self.learning_rate = learning_rate

    def forward(self, x: ndarray) -> None:
        self.x = x
        self.y = np.dot(x, self.W) + self.b

    def backward(self, dL_dy) -> ndarray:
        dL_dx = np.dot(dL_dy, self.W.T)
        dL_dW = np.dot(self.x.T, dL_dy)
        dL_db = np.sum(dL_dy, axis=0, keepdims=True)


        self.W -= self.learning_rate * dL_dW
        self.b -= self.learning_rate * dL_db

        return dL_dx


class SigmoidActivation:
    # Implementation of the sigmoid activation function, which
    # squashes values between 0 and 1.
    #
    # Formula: y = 1 / (1 + e^-x)

    def forward(self, x: ndarray) -> None:
        self.x = x
        x = np.clip(x, -500, 500)
        self.y = 1 / (1 + np.exp(-x))

    def backward(self, dL_dy: ndarray) -> ndarray:
        dy_dx = self.y * (1 - self.y)
        dL_dx = dL_dy * dy_dx

        return dL_dx


class ReLUActivation:
    # Implementation of the ReLu activation function, which captures
    # non-linearity by setting negative values to 0.
    #
    # Formula: y = max(0, x)

    def forward(self, x: ndarray) -> None:
        self.x = x
        self.y = np.maximum(0, x)

    def backward(self, dL_dy: ndarray) -> ndarray:
        dy_dx = (self.x > 0).astype(int)
        dL_dx = dL_dy * dy_dx

        return dL_dx


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
