"""
Train the neural network on the data and evaluate the model on the test data.

Notation:
    - x: inputs (array)
    - y: output (array)
"""

from numpy import ndarray
from data import Data, train_test_split
from neural_network import NeuralNetwork
import numpy as np

data = Data("./training_data/data.csv", drop_columns=["id"])
x_train, y_train, x_test, y_test = train_test_split(data.x, data.y, split=0.3)

def evaluate_model(trained_model: NeuralNetwork, x: ndarray, y_true: ndarray):
    """
    Evaluate the model on the test data.
    """
    y_pred = trained_model.forward(x)
    loss = trained_model.loss_func.forward(y_pred, y_true)
    return loss

def tune_hyperparams(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    n_layers_opts: list[int] = [2, 3, 4],
    n_neurons_opts: list[int] = [4, 8, 16],
    learning_rate_opts: ndarray = np.arange(0.005, 0.105, 0.005),
    n_epochs_opts: list[int] = range(100, 200, 10),
) -> dict:
    best_loss = float("inf")
    best_params = {}
    for n_layers in n_layers_opts:
        for n_neurons in n_neurons_opts:
            for i in range(len(learning_rate_opts)):
                for n_epochs in n_epochs_opts:
                    model = NeuralNetwork(
                        n_layers=n_layers,
                        n_inputs=len(x_train[0]),
                        n_outputs=1,
                        n_neurons=n_neurons,
                        hidden_act="ReLU",
                        output_act="Sigmoid",
                        loss_func="BCE",
                        learning_rate=learning_rate_opts[i],
                        n_epochs=n_epochs,
                    )
                    model.fit(x_train, y_train)

                    test_loss = evaluate_model(model, x_test, y_test)

                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_params = {
                            "n_layers": n_layers,
                            "n_neurons": n_neurons,
                            "learning_rate": learning_rate_opts[i],
                            "n_epochs": n_epochs,
                        }
                        print("New best params:", best_params)
                        print("New best test loss:", test_loss)
    return best_params

model = NeuralNetwork(
    n_layers=2,
    n_inputs=len(x_train[0]),
    n_outputs=1,
    n_neurons=16,
    hidden_act="ReLU",
    output_act="Sigmoid",
    loss_func="BCE",
    learning_rate=0.02,
    n_epochs=160,
)

model.fit(x_train, y_train)
test_loss = evaluate_model(model, x_test, y_test)
print("Test Loss:", test_loss)