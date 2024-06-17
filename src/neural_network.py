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
import shutil


class NeuralNetwork:
    """
    A neural network model that can be trained on data and used to make predictions.
    """

    def __init__(
        self,
        n_layers: int = 0,
        n_inputs: int = 0,
        n_outputs: int = 0,
        n_neurons: int = 0,
        hidden_act: str = None,
        output_act: str = None,
        loss_func: str = None,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        from_model: str = None,
    ):
        if from_model is not None:
            self.load(from_model)
            return
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
        if os.path.exists(name):
            shutil.rmtree(name)
        os.makedirs(name, exist_ok=True)
        architecture = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                w, b = layer.get_weights_biases()
                np.save(os.path.join(name, f'weights_{i}.npy'), w)
                np.save(os.path.join(name, f'biases_{i}.npy'), b)
                architecture.append({
                    'type': 'DenseLayer',
                    'n_inputs': w.shape[0],
                    'n_neurons': w.shape[1]
                })
            elif isinstance(layer, (ReLUActivation, SigmoidActivation, SoftmaxActivation)):
                architecture.append({
                    'type': layer.__str__()
                })

        architecture.append({
            "type": "loss_func",
            "func": self.loss_func.__str__(),
        })

        with open(os.path.join(name, 'architecture.json'), 'w') as f:
            json.dump(architecture, f, indent=4)

    def load(self, name: str):
        with open(os.path.join(name, "architecture.json"), "r") as f:
            architecture = json.load(f)
        
        self.layers = []
        for i, layer_info in enumerate(architecture):
            if layer_info["type"] == "DenseLayer":
                n_inputs = layer_info["n_inputs"]
                n_neurons = layer_info["n_neurons"]
                layer = DenseLayer(n_inputs, n_neurons)
                layer.set_weights(np.load(os.path.join(name, f"weights_{i}.npy")))
                layer.set_biases(np.load(os.path.join(name, f"biases_{i}.npy")))
                self.layers.append(layer)
            elif layer_info["type"] == "loss_func":
                self.loss_func = self._initializers()["loss"][layer_info["func"]]()
            else:
                self.layers.append(self._initializers()["activation"][layer_info["type"]]())
        
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
