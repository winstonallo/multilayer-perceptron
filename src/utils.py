from src.neural_network import NeuralNetwork
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


def evaluate_model(model: NeuralNetwork, x_test: ndarray, y_test: ndarray) -> float:
    """
    Evaluate the model on the test data and return the loss.
    """
    y_pred = model.predict(x_test)
    return model.loss_func.forward(y_pred, y_test)

def plot_prediction_histogram(ax, y_true: ndarray, y_pred: ndarray, threshold: float = 0.5):
    """
    Create a plot of histograms of the predicted probabilities for each true class.
    """
    y_pred_pos = y_pred[y_true.flatten() == 1]
    y_pred_neg = y_pred[y_true.flatten() == 0]

    ax.hist(y_pred_pos, bins=50, alpha=0.5, label="Positive Class")
    ax.hist(y_pred_neg, bins=50, alpha=0.5, label="Negative Class")

    ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Predicted Probabilities")
    ax.legend()
    ax.grid(True)

def plot_confusion_matrix(fig, ax, cm: ndarray, class_names: list):
    """
    Create a plot of the confusion matrix using Matplotlib.
    """
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_training_metrics(axes, loss_history: list, accuracy_history: list):
    """
    Create a plot of the loss and accuracy over the training duration.
    """
    axes[0].plot(loss_history, label="Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Over Epochs")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(accuracy_history, label="Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Over Epochs")
    axes[1].legend()
    axes[1].grid(True)

def plot_combined_metrics(y_true: ndarray, y_pred: ndarray, cm: ndarray, loss_history: list, accuracy_history: list, class_names: list, threshold: float = 0.5):
    """
    Combine multiple plots into a single fullscreen plot.
    """
    combined_fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    combined_fig.suptitle('Model Performance Metrics', fontsize=20)

    plot_prediction_histogram(axes[0, 0], y_true, y_pred, threshold)
    plot_confusion_matrix(combined_fig, axes[0, 1], cm, class_names)
    plot_training_metrics(axes[1, :], loss_history, accuracy_history)

    combined_fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
