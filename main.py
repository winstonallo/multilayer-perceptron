from neural_network import NeuronLayer
import numpy as np

weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
inputs = [[1.0, 2.0, 3.0, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]
biases = [2.0, 3.0, 0.5]


layer_outputs = NeuronLayer.batch(weights=weights, inputs=inputs, biases=biases)
print(layer_outputs)
