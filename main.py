import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from neural_network import forward_pass

inputs = [[1.0, 2.0, 3.0, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]

weights = [
    np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]),
    np.array([[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]),
]

biases = [[2.0, 3.0, 0.5], [-1, 2, -0.5]]

layers = forward_pass(inputs=np.array(inputs), weights=weights, biases=np.array(biases))

final_output = layers[-1].output()
print(final_output)


nnfs.init()
x, y = spiral_data(samples=100, classes=3)
plt.scatter(x[:, 0], x[:, 1])
plt.show()