import numpy as np
import matplotlib.pyplot as plt
from data import Data, train_test_split
from neural_network import NeuralNetwork
from numpy import ndarray
from tune import tune
from classification_metrics import ClassificationMetrics

def plot_prediction_histogram(y_true: ndarray, y_pred: ndarray, threshold: float = 0.5):
    """
    Plot histograms of the predicted probabilities for each true class.
    """
    plt.figure(figsize=(10, 6))

    # Predicted probabilities for each class
    y_pred_pos = y_pred[y_true.flatten() == 1]
    y_pred_neg = y_pred[y_true.flatten() == 0]

    # Plot histograms
    plt.hist(y_pred_pos, bins=50, alpha=0.5, label="Positive Class")
    plt.hist(y_pred_neg, bins=50, alpha=0.5, label="Negative Class")

    # Threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')

    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Histogram of Predicted Probabilities")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm: ndarray, class_names: list):
    """
    Plot the confusion matrix using Matplotlib.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Annotate each cell in the matrix with its count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = Data("./training_data/data.csv", drop_columns=["id"])
    # np.random.seed(15)
    x_train, y_train, x_test, y_test = train_test_split(data.x, data.y, split=0.2)

    best_params, results = tune(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, seed=15)

    model = NeuralNetwork(from_pretrained="best")

    y_pred = model.predict(data.x)

    performance = ClassificationMetrics(y_pred, data.y)
    cm = performance.confusion_matrix()
    print("Confusion matrix:\n", cm)
    
    print("Precision:", performance.precision())
    print("Recall:", performance.recall())
    print("F1:", performance.f1())
    model.plot_metrics()

    # Plot the histogram of predicted probabilities
    plot_prediction_histogram(data.y, y_pred)

    # Plot the confusion matrix
    plot_confusion_matrix(cm, class_names=["Negative", "Positive"])
