import numpy as np
from data import Data, train_test_split
from neural_network import NeuralNetwork
from numpy import ndarray
from tune import tune


if __name__ == "__main__":
    data = Data("./training_data/data.csv", drop_columns=["id"])
    x_train, y_train, x_test, y_test = train_test_split(data.x, data.y, split=0.2)


    best_params, results = tune(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    model = NeuralNetwork(from_pretrained="best")
