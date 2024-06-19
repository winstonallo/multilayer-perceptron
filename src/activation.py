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
        """
        Return the string representation of the activation function.
        """
        raise NotImplementedError()


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
        """
        Return the string representation of the activation function.
        """
        return "Sigmoid"


class ReLUActivation(Activation):
    """
    Implementation of the ReLU activation function, which captures
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
        """
        Return the string representation of the activation function.
        """
        return "ReLU"


class SoftmaxActivation(Activation):
    """
    Implementation of the Softmax activation function, which is used
    in the output layer of a neural network to produce probabilities
    by ensuring each output is in a (0, 1) range and the sum
    of all outputs equals 1.

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
        self.x = x
        exponential_values = np.exp(x - np.max(x, axis=1, keepdims=True))

        # After getting the exponent for each normalized value, we divide each of them
        # by the sum of all others in their respective row.
        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
        self.y = probabilities

        return self.y

    def backward(self, dl_dy: ndarray) -> ndarray:
        """
        Calculate the partial derivative of the loss with respect to the inputs.
        """
        batch_size, num_classes = self.y.shape
        dl_dx = np.zeros_like(self.y)

        for i in range(batch_size):
            y_i = self.y[i].reshape(-1, 1)
            jacobian_m = np.diagflat(y_i) - np.dot(y_i, y_i.T)
            dl_dx[i] = np.dot(jacobian_m, dl_dy[i])

        return dl_dx

    def __str__(self):
        """
        Return the string representation of the activation function.
        """
        return "Softmax"
