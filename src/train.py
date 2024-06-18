import numpy as np
from data import Data, train_test_split
from neural_network import NeuralNetwork
from numpy import ndarray
from tune import tune
from classification_metrics import ClassificationMetrics


if __name__ == "__main__":

    data = Data("./training_data/data.csv", drop_columns=["id"])
    # np.random.seed(15)
    x_train, y_train, x_test, y_test = train_test_split(data.x, data.y, split=0.2)

    best_params, results = tune(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, seed=15)

    model = NeuralNetwork(from_pretrained="best")

    y_pred = model.predict(data.x)

    performance = ClassificationMetrics(y_pred, data.y)
    print("Confusion matrix:\n", performance.confusion_matrix())
    print("Precision:", performance.precision())
    print("Recall:", performance.recall())
    print("F1:", performance.f1())
    model.plot_metrics()
