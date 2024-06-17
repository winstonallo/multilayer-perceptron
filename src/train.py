from data import Data, train_test_split
from neural_network import NeuralNetwork
from layers import DenseLayer, SigmoidActivation, ReLUActivation, SoftmaxActivation
from loss import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss
import matplotlib.pyplot as plt
from numpy import ndarray


data = Data("./training_data/data.csv", drop_columns=["id"])
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
