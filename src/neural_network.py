import numpy as np
from numpy import ndarray
from dense_layer import DenseLayer
from activation import SigmoidActivation, ReLUActivation, SoftmaxActivation
from loss import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
import os
import json
import shutil
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    A neural network model that can be trained on data and used to make predictions.
    """

    def __init__(
        self,
        n_inputs: int = 0,
        n_outputs: int = 0,
        n_layers: int = 2,
        n_neurons: int = 3,
        hidden_act: str = "ReLU",
        output_act: str = "Sigmoid",
        loss_func: str = "BCE",
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        from_pretrained: str = None,
    ):
        if from_pretrained is None:
            assert n_inputs > 0 and n_outputs > 0, "Please specify the number of inputs."
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
            self.loss_history = []
            self.accuracy_history = []
            self.trained = False
            self._build()
        else:
            self._load(from_pretrained)
            self.trained = True
            return

    def fit(self, x: ndarray, y: ndarray):
        """
        Fit the neural network model to the training data.
        """
        for _ in range(self.n_epochs):
            y_pred = self._forward(x)
            loss = self.loss_func.forward(y_pred, y)
            self.loss_history.append(loss)
            accuracy = self._calculate_accuracy(y_pred, y)
            self.accuracy_history.append(accuracy)
            self._backward()
        self.trained = True

    def save(self, name: str):
        """
        Save the trained model to ./{name}/. This allows you to later load
        the model for further use.
        """
        self._clear_or_create_directory(name)
        architecture = self._get_architecture(name)
        self._save_architecture(name, architecture)
        self._save_metrics(name)

    def predict(self, x: ndarray) -> ndarray:
        """
        Make predictions using the trained neural network model.
        """
        assert self.trained, "Model has not been trained yet."
        return self._forward(x)

    def training_metrics(self):
        """
        Plot the loss and accuracy over epochs.
        """
        return self.loss_history, self.accuracy_history

    def _build(self):
        self.layers.append(DenseLayer(self.n_inputs, self.n_neurons, self.learning_rate))
        self.layers.append(self.hidden_act)

        for i in range(self.n_layers - 2):
            self.layers.append(DenseLayer(self.n_neurons, self.n_neurons, self.learning_rate))
            self.layers.append(self.hidden_act)

        self.layers.append(DenseLayer(self.n_neurons, self.n_outputs, self.learning_rate))
        self.layers.append(self.output_act)

    def _forward(self, x: ndarray):
        """
        Make a forward pass through the neural network.
        """
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def _backward(self):
        """
        Make a backward pass through the neural network to update the weights.
        """
        dl_dy = self.loss_func.backward()
        for layer in reversed(self.layers):
            dl_dy = layer.backward(dl_dy)

    def _calculate_accuracy(self, y_pred: ndarray, y_true: ndarray) -> float:
        """
        Calculate the accuracy of the predictions.
        """
        if y_pred.shape[1] == 1:
            predictions = (y_pred > 0.5).astype(int).flatten()
            targets = y_true.flatten()
        else:
            predictions = np.argmax(y_pred, axis=1)
            targets = np.argmax(y_true, axis=1)
    
        accuracy = np.mean(predictions == targets)
        return accuracy

    def _load(self, name: str):
        architecture = self._load_architecture(name)
        self._load_metrics(name)

        self.layers = []
        for i, layer_info in enumerate(architecture):
            if layer_info["type"] == "DenseLayer":
                self._load_dense_layer(layer_info, i, name)

            elif layer_info["type"] == "loss_func":
                self.loss_func = self._initializers()["loss"][layer_info["func"]]()
            else:
                self.layers.append(self._initializers()["activation"][layer_info["type"]]())

    def _load_dense_layer(self, layer_info: dict, index: int, name: str):
        layer = DenseLayer(layer_info["n_inputs"], layer_info["n_neurons"])
        layer.set_weights(np.load(os.path.join(name, f"weights_{index}.npy")))
        layer.set_biases(np.load(os.path.join(name, f"biases_{index}.npy")))
        self.layers.append(layer)

    def _load_architecture(self, name: str):
        with open(os.path.join(name, "architecture.json"), "r") as f:
            architecture = json.load(f)
        return architecture

    def _get_architecture(self, name: str):
        architecture = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                architecture.append(self._get_dense_layer_info(i, layer, name))
            elif isinstance(layer, (ReLUActivation, SigmoidActivation, SoftmaxActivation)):
                architecture.append({"type": layer.__str__()})
        architecture.append({"type": "loss_func", "func": self.loss_func.__str__()})
        return architecture

    def _get_dense_layer_info(self, i: int, layer: DenseLayer, name: str):
        w, b = layer.get_weights_biases()
        np.save(os.path.join(name, f"weights_{i}"), w)
        np.save(os.path.join(name, f"biases_{i}"), b)
        return {"type": "DenseLayer", "n_inputs": w.shape[0], "n_neurons": w.shape[1]}

    def _save_architecture(self, name: str, architecture: list[dict]):
        with open(os.path.join(name, "architecture.json"), "w") as f:
            json.dump(architecture, f, indent=4)

    def _save_metrics(self, name: str):
        with open(os.path.join(name, "loss_history.json"), "w") as f:
            json.dump(self.loss_history, f, indent=4)
        with open(os.path.join(name, "accuracy_history.json"), "w") as f:
            json.dump(self.accuracy_history, f, indent=4)

    def _load_metrics(self, name: str):
        with open(os.path.join(name, "loss_history.json"), "r") as f:
            self.loss_history = json.load(f)
        with open(os.path.join(name, "accuracy_history.json"), "r") as f:
            self.accuracy_history = json.load(f)

    def _clear_or_create_directory(self, name: str):
        if os.path.exists(name):
            shutil.rmtree(name)
        os.makedirs(name, exist_ok=True)

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
