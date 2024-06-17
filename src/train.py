from data import Data, train_test_split
from layers import DenseLayer, SigmoidActivation, ReLUActivation, SoftmaxActivation
from loss import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
import matplotlib.pyplot as plt
from numpy import ndarray


class NeuralNetwork:

    def __init__(
        self,
        n_layer: int,
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
        self.n_layer = n_layer
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.hidden_act = self.initializers()["activation"][hidden_act]()
        self.output_act = self.initializers()["activation"][output_act]()
        self.loss_func = self.initializers()["loss"][loss_func]()
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
        losses = []
        accuracies = []
        plt.ion()
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss", color="tab:red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="tab:blue")

        for epoch in range(self.n_epochs):
            y_pred = self.forward(x)
            L = self.loss_func.forward(y_pred, y_true)
            losses.append(L)

            accuracy = ((y_pred > 0.5) == y_true).mean()
            accuracies.append(accuracy)

            self.backward()

            ax1.plot(losses, color="tab:red", label="Loss" if epoch == 0 else "")
            ax2.plot(accuracies, color="tab:blue", label="Accuracy" if epoch == 0 else "")
            plt.draw()
            plt.pause(0.05)

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.ioff()
        plt.show()

    def forward(self, x: ndarray):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward(self):
        dL_dy = self.loss_func.backward()
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)

    def initializers(self):
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


data = Data("./data/raw/data.csv", drop_columns=["id"])
x_train, y_train, x_test, y_test = train_test_split(data.X, data.y, split=0.3)

model = NeuralNetwork(
    n_layer=3,
    n_inputs=len(x_train[0]),
    n_outputs=1,
    n_neurons=4,
    hidden_act="ReLU",
    output_act="Sigmoid",
    loss_func="BCE",
    learning_rate=0.01,
    n_epochs=150,
)

model.fit(x_train, y_train)


def evaluate_model(model: NeuralNetwork, x: ndarray, y_true: ndarray):
    y_pred = model.forward(x)
    loss = model.loss_func.forward(y_pred, y_true)
    return loss


test_loss = evaluate_model(model, x_test, y_test)
print("Test Loss:", test_loss)
