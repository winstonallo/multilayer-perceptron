import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from neural_network import DenseLayer, ReLuActivation, SoftMaxActivation

nnfs.init()
x, y = spiral_data(samples=100, classes=3)

l1 = DenseLayer(2, 3)
a1 = ReLuActivation()
l2 = DenseLayer(3, 3)
a2 = SoftMaxActivation()

l1.forward(x)
a1.forward(l1.output)

l2.forward(a1.output)
a2.forward(l2.output)

print(a2.output)
