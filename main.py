from neural_network import NeuralNetwork

weights1 = [2, 0, 0]  # First neuron's weights
weights2 = [2, 2, 0]  # Second neuron's weights
weights3 = [2, 2, 2]  # Third neuron's weights

neural_network = NeuralNetwork(weights=[weights1, weights2, weights3], biases=[0, 0, 1])

outputs = neural_network.run(inputs=[1.1, 2.2, 3.3])
print(outputs)
