import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from neural_network import DenseLayer, ReLUActivation, SoftmaxActivation

nnfs.init()
x, y = spiral_data(samples=100, classes=3)

l1 = DenseLayer(2, 3)
a1 = ReLUActivation()
l2 = DenseLayer(3, 3)
a2 = SoftmaxActivation()

l1.forward(x)
a1.forward(l1.output)

l2.forward(a1.output)
a2.forward(l2.output)

print(a2.output[:5])

import numpy as np

one_hot_encoding = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
)

predictions = np.array(
    [
        [0.7, 0.1, 0.2],
        [0.1, 0.5, 0.4],
        [0.02, 0.9, 0.08],
    ]
)

A = one_hot_encoding * predictions
B = np.sum(A, axis=1)
C = -np.log(B)
loss = np.mean(C)
print(loss)
