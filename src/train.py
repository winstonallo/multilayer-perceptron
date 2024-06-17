"""
Train the neural network on the data and evaluate the model on the test data.

Notation:
    - x: inputs (array)
    - y: output (array)
"""

from numpy import ndarray
from data import Data, train_test_split
from neural_network import NeuralNetwork


data = Data("./training_data/data.csv", drop_columns=["id"])
x_train, y_train, x_test, y_test = train_test_split(data.x, data.y, split=0.3)

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


def evaluate_model(trained_model: NeuralNetwork, x: ndarray, y_true: ndarray):
    """
    Evaluate the model on the test data.
    """
    y_pred = trained_model.forward(x)
    loss = trained_model.loss_func.forward(y_pred, y_true)
    return loss


test_loss = evaluate_model(model, x_test, y_test)
print("Test Loss:", test_loss)
