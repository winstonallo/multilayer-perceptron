from data import TrainingData
from layers import DenseLayer, SigmoidActivation, ReLUActivation
from loss import BinaryCrossEntropyLoss
import matplotlib.pyplot as plt


# Notation:
#   - x: inputs (array)
#   - y: output (array)
#   - W: weights (array)
#   - b: biases (array)
#   - l: neuron layer
#   - a: activation layer
#   - L: loss function


class Trainer:

    def __init__(self):
        self.x, self.y_true = TrainingData().get_data()


trainer = Trainer()


inputs = trainer.x
targets = trainer.y_true

l1 = DenseLayer(len(inputs[0]), 3, 0.01)
a1 = ReLUActivation()

l2 = DenseLayer(3, 3, 0.01)
a2 = ReLUActivation()

l3 = DenseLayer(3, 1, 0.01)
a3 = SigmoidActivation()

L = BinaryCrossEntropyLoss()

losses = []

for i in range(200):

    l1.forward(inputs)
    a1.forward(l1.y)

    l2.forward(a1.y)
    a2.forward(l2.y)

    l3.forward(a2.y)
    a3.forward(l3.y)

    loss = L.calculate(a3.y, targets)
    losses.append(loss)

    dL_dy = L.backward()

    dL_dx = a3.backward(dL_dy)
    dL_dx = l3.backward(dL_dx)

    dL_dx = a2.backward(dL_dy)
    dL_dx = l2.backward(dL_dx)

    dL_dx = a1.backward(dL_dx)
    dL_dx = l1.backward(dL_dx)

    # if i % 20 == 0:
    print(f"Iteration no.{i + 1}: Loss = {loss}")
    print(f"Output: {a3.y}")


plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
# plt.show()