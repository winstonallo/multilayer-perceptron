from neural_network import NeuronLayer
import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]

l1_weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
l1_biases = [2.0, 3.0, 0.5]

l1 = NeuronLayer(weights=l1_weights, inputs=inputs, biases=l1_biases)

l2_weights = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
l2_biases = [-1, 2, -0.5]

l2 = NeuronLayer(weights=l2_weights, inputs=l1.output(), biases=l2_biases)


output = l2.output()
print(output)

