from neural_network import NeuralNetwork
import numpy as np
from numpy import ndarray


def evaluate_model(model: NeuralNetwork, x_test: ndarray, y_test: ndarray) -> float:
    """
    Evaluate the model on the test data and return the loss.
    """
    y_pred = model.predict(x_test)
    return model.loss_func.forward(y_pred, y_test)
