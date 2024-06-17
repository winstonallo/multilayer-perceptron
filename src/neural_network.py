"""
This module contains the implementation of a neural network. The neural network 
is currently limited to the following configurations:
- Layers: DenseLayer
- Activation functions: ReLUActivation, SigmoidActivation, SoftmaxActivation
- Loss functions: BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
"""

import matplotlib.pyplot as plt
from numpy import ndarray
from layers import DenseLayer, SigmoidActivation, ReLUActivation, SoftmaxActivation
from loss import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss


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
        early_stopping: bool = True,
        patience: int = 10,
    ):
        self.n_layer = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.hidden_act = self._initializers()["activation"][hidden_act]()
        self.output_act = self._initializers()["activation"][output_act]()
        self.loss_func = self._initializers()["loss"][loss_func]()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.layers = []
        self.early_stopping = early_stopping
        self.patience = patience
        self._build()

    def _build(self):
        # Input layer
        self.layers.append(DenseLayer(self.n_inputs, self.n_neurons, self.learning_rate))
        self.layers.append(self.hidden_act)

        # Hidden layers
        for _ in range(self.n_layer - 2):
            self.layers.append(DenseLayer(self.n_neurons, self.n_neurons, self.learning_rate))
            self.layers.append(self.hidden_act)

        # Output layer
        self.layers.append(DenseLayer(self.n_neurons, self.n_outputs, self.learning_rate))
        self.layers.append(self.output_act)

    def fit(self, x: ndarray, y_true: ndarray):
        """
        Fit the neural network model to the training data.
        """
        losses = []
        accuracies = []
        # plt.ion()
        # _, ax1 = plt.subplots()

        # ax1.set_xlabel("Iteration")
        # ax1.set_ylabel("Loss", color="tab:red")
        # ax2 = ax1.twinx()
        # ax2.set_ylabel("Accuracy", color="tab:blue")

        for epoch in range(self.n_epochs):
            y_pred = self.forward(x)
            loss = self.loss_func.forward(y_pred, y_true)
            losses.append(loss)

            accuracy = ((y_pred > 0.5) == y_true).mean()
            accuracies.append(accuracy)

            self.backward()

            # ax1.plot(losses, color="tab:red", label="Loss" if epoch == 0 else "")
            # ax2.plot(accuracies, color="tab:blue", label="Accuracy" if epoch == 0 else "")
            # plt.draw()
            # plt.pause(0.05)

        # ax1.legend(loc="upper left")
        # ax2.legend(loc="upper right")
        # plt.ioff()
        # plt.show()

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
