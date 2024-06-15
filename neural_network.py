import numpy as np
from numpy import ndarray


class Neuron:
    # The functionality of this class is abstracted away in NeuronLayer
    # using the dot product of the weights and the inputs.

    def output(self, weights: list[float], inputs: list[float], bias: float) -> float:
        # Summation of weighted inputs with added bias.
        return np.dot(inputs, weights) + bias


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


class Loss:
    # Base class for losses. Gets the array of losses from the forward()
    # method implemented in the child and returns its mean, which will be our
    # final value.

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        loss = np.mean(sample_losses)
        return loss


class CategoricalCrossEntropyLoss(Loss):
    # Categorical Cross Entropy loss takes the negative logarithm of
    # the relevant prediction - i.e the one matching the expected
    # output - as a precision indicator.
    # The negative logarithm crosses the y-axis at x = 1, which enables
    # us to have the expected relationship between the confidence 
    # and the loss (higher confidence -> lower loss and vice versa).

    def forward(self, prediction: ndarray, true_value: ndarray):
        # Number of samples in the batch.
        samples = len(prediction)

        # Clip data to prevent division by zero and perfect outputs
        # (1), which are not good for gradients.
        clipped_prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

        if len(true_value.shape) == 1:
            # If this is true, the true values are just a 1D array,
            # for example: [0, 1, 1] ([RED, GREEN, BLUE]).
            # These numbers stand for the indexes of the true value in
            # each row of the prediction. Therefore, we use python's
            # smart indexing in order to choose which indices of the
            # predictions are to be taken into account.
            confidence = clipped_prediction[range(samples), true_value]

        elif len(true_value.shape) == 2:
            # If this is true, the true values are a one hot encoded array.
            # Multiplying the one hot encoded array with the predictions
            # will zero out the false predictions, thus ignoring them
            # in loss calculation.
            confidence = np.sum(clipped_prediction * true_value, axis=1)

        negative_log_likelihoods = -np.log(confidence)
        # The return array consists of the negative logs of the confidence
        # for all correct predictions. This is what we will take the mean
        # of to calculate the final loss.

        return negative_log_likelihoods
