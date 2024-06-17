"""
This module contains the implementation of a neural network. The neural network 
is currently limited to the following configurations:
- Layers: DenseLayer
- Activation functions: ReLUActivation, SigmoidActivation, SoftmaxActivation
- Loss functions: BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from layers import DenseLayer, SigmoidActivation, ReLUActivation, SoftmaxActivation
from loss import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
import os
import json


class NeuralNetwork:
    """
    A neural network model that can be trained on data and used to make predictions.
    """

    def __init__(
        self,
        n_layers: int,
        n_inputs: int,
        n_outputs: int,
        n_neurons: int,
        hidden_act: str,
        output_act: str,
        loss_func: str,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        from_model: str = None
    ):
        if from_model is not None:
            pass # self.load(...)
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.hidden_act = self._initializers()["activation"][hidden_act]()
        self.output_act = self._initializers()["activation"][output_act]()
        self.loss_func = self._initializers()["loss"][loss_func]()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.layers = []
        self._build()

    def _build(self):
        self.layers.append(DenseLayer(self.n_inputs, self.n_neurons, self.learning_rate))
        self.layers.append(self.hidden_act)

        for _ in range(self.n_layers - 2):
            self.layers.append(DenseLayer(self.n_neurons, self.n_neurons, self.learning_rate))
            self.layers.append(self.hidden_act)

        self.layers.append(DenseLayer(self.n_neurons, self.n_outputs, self.learning_rate))
        self.layers.append(self.output_act)

    def fit(self, x: ndarray, y_true: ndarray):
        """
        Fit the neural network model to the training data.
        """
        losses = []
        accuracies = []

        for epoch in range(self.n_epochs):
            y_pred = self.forward(x)
            loss = self.loss_func.forward(y_pred, y_true)
            losses.append(loss)

            accuracy = ((y_pred > 0.5) == y_true).mean()
            accuracies.append(accuracy)

            self.backward()

    def save(self, name: str):
        os.makedirs(name, exist_ok=True)
        weights = []
        biases = []
        architecture = {
            "n_layers": self.n_layers,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "n_neurons": self.n_neurons,
            "hidden_act": self.hidden_act.__str__(),
            "output_act": self.output_act.__str__(),
        }

        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                w, b = layer.get_weights_biases()
                weights.append(w)
                biases.append(b)

        for i, weights_set in enumerate(weights):
            np.save(os.path.join(name, f"weights_{i}.npy"), weights_set)
        for i, biases_set in enumerate(biases):
            np.save(os.path.join(name, f"biases_{i}.npy"), biases_set)
        with open(os.path.join(name, "architecture.json"), "w") as f:
            json.dump(architecture, f)
        
        print("Saved new best model in ./best/")

    def forward(self, x: ndarray):
        """
        Make a forward pass through the neural network.
        """
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward(self):
        """
        Make a backward pass through the neural network to update the weights.
        """
        dl_dy = self.loss_func.backward()
        for layer in reversed(self.layers):
            dl_dy = layer.backward(dl_dy)

    def _initializers(self):
        return {
            "activation": {
                "ReLU": ReLUActivation,
                "Sigmoid": SigmoidActivation,
                "Softmax": SoftmaxActivation,
            },
            "loss": {
                "BCE": BinaryCrossEntropyLoss,
                "CCE": CategoricalCrossEntropyLoss,
            },
        }
