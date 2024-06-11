from neuron import Neuron


class NeuralNetwork:

    def __init__(self, weights: list[list[float]], biases: list[float]) -> None:
        self.neurons = self.init_neurons(weights, biases)

    def init_neurons(self, weights: list[list[float]], biases: list[float]):
        # Create a list of Neuron objects, each with their own weights and biases.
        return [Neuron(weight_list, bias) for weight_list, bias in zip(weights, biases)]

    def run(self, inputs: list[float]) -> list[float]:
        # Get the outputs of all neurons using the inputs passed as a parameter.
        return [neuron.output(inputs) for neuron in self.neurons]
