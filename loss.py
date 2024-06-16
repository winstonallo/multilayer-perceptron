import numpy as np
from numpy import ndarray


class Loss:
    # Base class for losses. Gets the array of losses from the forward()
    # method implemented in the child and returns its mean, which will be our
    # final value.

    def calculate(self, y_pred: ndarray, y_true: ndarray):
        sample_losses = self.forward(y_pred, y_true)
        loss = np.mean(sample_losses)
        return loss


class BinaryCrossEntropyLoss(Loss):
    # Binary Cross Entropy loss takes the negative logarithm of
    # the relevant prediction - i.e the one matching the expected
    # output - as a precision indicator.
    # It is the simpler version of Categorical Cross Entropy Loss,
    # used when the output is binary. We do not need to sum the
    # losses, as there is only one output.
    #
    # Notations:
    #   - BCE: Binary Cross Entropy
    #   - y_true: True values
    #   - y_pred: Predicted values
    #   - c: Complement (1 - val)
    #
    # Formula:
    #   BCE = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    def calculate(self, predictions: ndarray, targets: ndarray) -> float:
        # Clip the predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.predictions = predictions
        self.targets = targets
        
        # Calculate binary cross-entropy loss
        loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return loss

    def backward(self) -> ndarray:
        # Compute the gradient of the loss with respect to predictions
        dL_dy = - (self.targets / self.predictions) + ((1 - self.targets) / (1 - self.predictions))
        return dL_dy


class CategoricalCrossEntropyLoss(Loss):
    # Categorical Cross Entropy loss takes the negative logarithm of
    # the relevant prediction - i.e the one matching the expected
    # output - as a precision indicator.
    # The negative logarithm crosses the y-axis at x = 1, which enables
    # us to have the expected relationship between the confidence
    # and the loss (higher confidence -> lower loss and vice versa).
    #
    # Notations:
    #   - CCE: Categorical Cross Entropy
    #   - y_true: True values
    #   - y_pred: Predicted values
    #
    # Formula:
    #   CCE = -log(y_pred[correct_index])

    def forward(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        # Number of samples in the batch.
        samples = len(y_pred)

        # Clip data to prevent division by zero and perfect outputs
        # (1), which are not good for gradients.
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # If this is true, the true values are just a 1D array,
            # for example: [0, 1, 1] ([RED, GREEN, BLUE]).
            # These numbers stand for the indexes of the true value in
            # each row of the prediction. Therefore, we use python's
            # smart indexing in order to choose which indices of the
            # predictions are to be taken into account.
            confidence = y_pred_clip[range(samples), y_true]

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

        return negative_log_likelihoods
