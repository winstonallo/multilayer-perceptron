import numpy as np


class Neuron:
    # The functionality of this class is abstracted away in NeuronLayer
    # using the dot product of the weights and the inputs.

    def output(self, weights: list[float], inputs: list[float], bias: float) -> float:
        # Summation of weighted inputs with added bias.
        return np.dot(inputs, weights) + bias


class NeuronLayer:

    def __init__(self, weights: np.ndarray, inputs: np.ndarray, biases: np.ndarray):
        self.weights = np.array(weights)
        self.inputs = np.array(inputs)
        self.biases = np.array(biases)

    def run(self, weights: np.ndarray, inputs: np.ndarray, biases: np.ndarray) -> np.ndarray:
        # Get the outputs of the entire layer in one line using the dot product.
        # In this case, np.dot() multiplies the weights matrix (X x Y)
        # with the inputs vector (1 x X).
        # See the Neuron class.
        return np.dot(weights, inputs) + biases

    def output(self) -> np.ndarray:
        # This gets the layer outputs for a whole batch of inputs at a time.
        # Inputs and weights are both X x Y, therefore we must transpose the weights
        # in order to convert them into Y x X and make the dot product possible.
        return np.dot(self.inputs, self.weights.T) + self.biases


def forward_pass(inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> list[NeuronLayer]:
    passes = len(weights)
    layers = []
    for i in range(passes):
        if i == 0:
            layers.append(NeuronLayer(weights=weights[i], inputs=inputs, biases=biases[i]))
        else:
            layers.append(NeuronLayer(weights=weights[i], inputs=layers[i - 1].output(), biases=biases[i]))
    return layers
