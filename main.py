from neural_network import NeuralNetwork

neural_network = NeuralNetwork(
    weights=[[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]],
    biases=[2, 3, 0.5],
)

outputs = neural_network.run(inputs=[1, 2, 3, 2.5])
print(outputs)
