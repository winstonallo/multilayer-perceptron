import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from neural_network import DenseLayer, ReLUActivation, SoftmaxActivation, CategoricalCrossEntropyLoss

nnfs.init()
x, y = spiral_data(samples=100, classes=3)

l1 = DenseLayer(2, 3)
a1 = ReLUActivation()
l2 = DenseLayer(3, 3)
a2 = SoftmaxActivation()
loss_function = CategoricalCrossEntropyLoss()

l1.forward(x)
a1.forward(l1.output)

l2.forward(a1.output)
a2.forward(l2.output)

print(a2.output[:5])

loss = loss_function.calculate(a2.output, y)

print("loss", loss)

import numpy as np
predictions = np.argmax(a2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print("accuracy:", accuracy)