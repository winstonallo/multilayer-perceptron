from data import TrainingData, TestData
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
        plt.ion()
        plt.figure()
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        for epoch in range(self.n_epochs):
            y_pred = self.forward(x)
            L = self.loss_func.calculate(y_pred, y_true)
            losses.append(L)

            # Implement early stopping here
            self.backward()

            plt.cla()
            plt.plot(losses)
            plt.draw()
            plt.pause(0.01)

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


x_train, y_train = TrainingData().get_data()
x_test, y_test = TestData().get_data()

model = NeuralNetwork(
    n_layer=3,
    n_inputs=len(x_train[0]),
    n_outputs=1,
    n_neurons=3,
    hidden_act="ReLU",
    output_act="Sigmoid",
    loss_func="BCE",
    learning_rate=0.01,
    n_epochs=100,
)

model.fit(x_train, y_train)


def evaluate_model(model: NeuralNetwork, x: ndarray, y_true: ndarray):
    y_pred = model.forward(x)
    loss = model.loss_func.calculate(y_pred, y_true)
    return loss


test_loss = evaluate_model(model, x_test, y_test)
print("Test Loss:", test_loss)
