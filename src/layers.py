"""
The following classes are used to create layers in a neural network.
Each class has a forward method that calculates the output of the layer
based on the input it receives.

Notation:
  - x: inputs (array)
  - y: output (array)
  - w: weights (array)
  - b: biases (array)
  - n_{x | neurons}: number of {x | neurons}
  - l: loss
  - d{1}_d{2}: partial derivative of {1} with respect to {2}
"""

import numpy as np
from numpy import ndarray


class DenseLayer:
    """
    A dense layer in the neural network, which performs a linear transformation
    on the input data.
    """

    def __init__(self, n_x: int, n_neurons: int, learning_rate: float):
        self.w = 0.01 * np.random.randn(n_x, n_neurons) / np.sqrt(n_x)
        self.b = np.zeros((1, n_neurons))
        self.learning_rate = learning_rate
        self.x = None
        self.y = None

    def forward(self, x: ndarray) -> ndarray:
        """
        Calculate the weighted sum with added bias (output) for the layer.
        """
        self.x = x
        self.y = np.dot(x, self.w) + self.b

        return self.y

    def backward(self, dl_dy) -> ndarray:
        """
        Calculate the partial derivatives of the loss with respect to the weights,
        biases, and inputs of the layer.
        """
        dl_dx = np.dot(dl_dy, self.w.T)
        dl_dw = np.dot(self.x.T, dl_dy)
        dl_db = np.sum(dl_dy, axis=0, keepdims=True)

        self.w -= self.learning_rate * dl_dw
        self.b -= self.learning_rate * dl_db

        return dl_dx

    def get_weights_biases(self):
        return self.w, self.b


class Activation:
    """
    An abstract class for activation functions.
    """

    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x: ndarray) -> ndarray:
        """
        Calculate the activation function output.
        """
        raise NotImplementedError()

    def backward(self, dl_dy: ndarray) -> ndarray:
        """
        Calculate the partial derivative of the loss with respect to the inputs.
        """
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError


class SigmoidActivation(Activation):
    """
    Implementation of the sigmoid activation function, which
    squashes values between 0 and 1.

    Formula: y = 1 / (1 + e^-x)
    """

    def forward(self, x: ndarray) -> ndarray:
        """
        Calculate the sigmoid activation function output.
        """
        self.x = x
        x = np.clip(x, -500, 500)
        self.y = 1 / (1 + np.exp(-x))

        return self.y

    def backward(self, dl_dy: ndarray) -> ndarray:
        """
        Calculate the partial derivative of the loss with respect to the inputs.
        """
        dy_dx = self.y * (1 - self.y)
        dl_dx = dl_dy * dy_dx

        return dl_dx

    def __str__(self):
        return "Sigmoid"


class ReLUActivation(Activation):
    """
    Implementation of the ReLu activation function, which captures
    non-linearity by setting negative values to 0.

    Formula: y = max(0, x)
    """

    def forward(self, x: ndarray) -> ndarray:
        """
        Calculate the ReLU activation function output.
        """
        self.x = x
        self.y = np.maximum(0, x)

        return self.y

    def backward(self, dl_dy: ndarray) -> ndarray:
        """
        Calculate the partial derivative of the loss with respect to the inputs.
        """
        dy_dx = (self.x > 0).astype(int)
        dl_dx = dl_dy * dy_dx

        return dl_dx

    def __str__(self):
        return "ReLU"


class SoftmaxActivation(Activation):
    """
    Implementation of the Softmax activation function, which is used
    in the output layer of a neural network to produce probabilities
    by ensuring each output is in a (0, 1) range and the sum
    of all outputs equals to 0.

    Formula: e^x / sum(e^x)
    """

    def forward(self, x: ndarray) -> ndarray:
        """
        For each row, we subtract its highest value from all others before
        taking the exponents. This prevents the exponents from getting
        too big.
        We use keepdims=True in order to keep the dimensions of the original array.
        This is VERY important, as mismatched dimensions lead to unrelated values
        being subtracted from the rows.
        """
        exponential_values = np.exp(x - np.max(x, axis=1, keepdims=True))

        # After getting the exponent for each normalized value, we divide each of them
        # by the sum of all others in their respective row.
        #
        # Example, for this array: [x y z]
        # x = e^x / e^x + e^y + e^z
        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
        self.y = probabilities

        return self.y

    def backward(self, dl_dy: ndarray) -> ndarray:
        """
        Calculate the partial derivative of the loss with respect to the inputs.
        """
        raise NotImplementedError("Softmax backward pass is not implemented.")

    def __str__(self):
        return "Softmax"
