"""
This module contains the implementation of the DenseLayer class, which
performs a linear transformation on the input data. The DenseLayer class
is used in the neural network to connect the input data to the output
layer.

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

    def __init__(self, n_x: int, n_neurons: int, learning_rate: float = None):
        self.w = 0.01 * np.random.randn(n_x, n_neurons) / np.sqrt(n_x)
        self.b = np.zeros((1, n_neurons))
        self.learning_rate = learning_rate
        self.x = None
        self.y = None

    def set_weights(self, weights: ndarray) -> None:
        self.w = weights

    def set_biases(self, biases: ndarray) -> None:
        self.b = biases

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
