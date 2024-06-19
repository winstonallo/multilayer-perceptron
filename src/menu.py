from src.data import Data, train_test_split
from src.neural_network import NeuralNetwork
from src.tune import tune
from src.classification_metrics import ClassificationMetrics
from src.utils import plot_combined_metrics
from simple_term_menu import TerminalMenu
import sys
import os

if __name__ == "__main__":
    data = Data("./training_data/data.csv", drop_columns=["id"])
    x_train, y_train, x_test, y_test = train_test_split(data.x, data.y, split=0.2)

    best_params, results = tune(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, seed=15)

    model = NeuralNetwork(from_pretrained="best")

    y_pred = model.predict(data.x)

    performance = ClassificationMetrics(y_pred, data.y)
    cm = performance.confusion_matrix()
    print("Precision:", performance.precision())
    print("Recall:", performance.recall())
    print("F1:", performance.f1())

    loss_history, accuracy = model.training_metrics()

    plot_combined_metrics(data.y, y_pred, cm, loss_history, accuracy, ["Negative", "Positive"])


class Menu:

    BASE_OPTS = ["Train new model", "Use pre-trained model", "Quit"]
    TRAINING_OPTS = ["User-defined parameters", "Optimized parameters", "Go back"]

    def run(self):
        while True:
            self._clear()
            idx = self._base_menu()
            self._funcs()[self.BASE_OPTS[idx]]()

    def _base_menu(self) -> int:
        menu = TerminalMenu(self.BASE_OPTS)
        idx = menu.show()
        return idx

    def _train(self):
        self._clear()
        menu = TerminalMenu(self.TRAINING_OPTS)
        idx = menu.show()

    def _from_pretrained(self):
        print("From pretrained")

    def _user_defined_params(self):
        print("User defined")

    def _optimized_params(self):
        print("Optimized")

    def _clear(self):
        os.system("clear")

    def _exit(self):
        sys.exit(0)

    def _funcs(self):
        return {
            "Train new model": self._train,
            "Use pre-trained model": self._from_pretrained,
            "User-defined parameters": self._user_defined_params,
            "Optimized parameters": self._optimized_params,
            "Quit": self._exit,
        }
