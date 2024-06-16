from data import TrainingData
from layers import DenseLayer, SigmoidActivation


class Trainer:

    def __init__(self):
        self.inputs, self.targets = TrainingData().get_data()


trainer = Trainer()


inputs = trainer.inputs

layer = DenseLayer(len(inputs[0]), 3)
print(f"Initial weights: {layer.W}")