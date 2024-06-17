"""
This module contains the implementation of activation functions
used in the neural network. Activation functions are used to
introduce non-linearity to the network, allowing it to learn
complex patterns in the data.

The following activation functions are implemented:
    - SigmoidActivation
    - ReLUActivation
    - SoftmaxActivation

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