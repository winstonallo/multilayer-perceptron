"""
This module contains the loss functions used for error calculation
in the neural network. They are used to calculate the difference
between the predicted values and the true values.
"""

import numpy as np
from numpy import ndarray


class BinaryCrossEntropyLoss:
    """
    Binary Cross Entropy loss takes the negative logarithm of
    the relevant prediction - i.e the one matching the expected
    output - as a precision indicator.
    It is the simpler version of Categorical Cross Entropy Loss,
    used when the output is binary. We do not need to sum the
    losses, as there is only one output.

    Notations:
        - BCE: Binary Cross Entropy
        - y_true: True values
        - y_pred: Predicted values
        - c: Complement (1 - val)

    Formula:
        BCE = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    """

    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions: ndarray, targets: ndarray) -> float:
        """
        This method calculates the binary cross-entropy loss
        between the predictions and the targets.
        We clip the predictions to prevent division by zero and
        perfect outputs (1), which are not good for gradients.
        """
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        self.predictions = predictions
        self.targets = targets

        loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

        return loss

    def backward(self) -> ndarray:
        """
        This method calculates the derivative of the loss with
        respect to the predictions. It is used in backpropagation
        to calculate the gradients of the loss with respect to
        the weights and biases.
        """
        dl_dy = -(self.targets / self.predictions) + ((1 - self.targets) / (1 - self.predictions))
        return dl_dy


class CategoricalCrossEntropyLoss:
    """
    Categorical Cross Entropy loss takes the negative logarithm of
    the relevant prediction - i.e the one matching the expected
    output - as a precision indicator.
    The negative logarithm crosses the y-axis at x = 1, which enables
    us to have the expected relationship between the confidence
    and the loss (higher confidence -> lower loss and vice versa).

    Notations:
        - CCE: Categorical Cross Entropy
        - y_true: True values
        - y_pred: Predicted values

    Formula:
        CCE = -log(y_pred[correct_index])
    """

    def forward(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        """
        This method calculates the categorical cross-entropy loss
        between the predictions and the targets.
        """
        n_samples = len(y_pred)

        # Prevent division by zero and perfect outputs
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)

        confidence = None

        if len(y_true.shape) == 1:
            # If this is true, the true values are just a 1D array,
            # for example: [0, 1, 1] ([RED, GREEN, BLUE]).
            # These numbers stand for the indexes of the true value in
            # each row of the prediction. Therefore, we use python's
            # smart indexing in order to choose which indices of the
            # predictions are to be taken into account.
            confidence = y_pred_clip[range(n_samples), y_true]

        elif len(y_true.shape) == 2:
            # If this is true, the true values are a one hot encoded array.
            # Multiplying the one hot encoded array with the predictions
            # will zero out the false predictions, thus ignoring them
            # in loss calculation.
            confidence = np.sum(y_pred_clip * y_true, axis=1)

        negative_log_likelihoods = -np.log(confidence)
        # The return array consists of the negative logs of the confidence
        # for all correct predictions. This is what we will take the mean
        # of to calculate the final loss.

        return np.mean(negative_log_likelihoods)

    def backward(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        """
        This method calculates the derivative of the loss with
        respect to the predictions. It is used in backpropagation
        to calculate the gradients of the loss with respect to
        the weights and biases.
        """

        # If the true values are a 1D array, we need to convert them
        # to a one hot encoded array.
        if len(y_true.shape) == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        # Calculate the gradient of the loss with respect to the predictions.
        grad = y_pred - y_true
        return grad
