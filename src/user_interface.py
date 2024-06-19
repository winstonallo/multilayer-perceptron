from src.data import Data, train_test_split
from src.neural_network import NeuralNetwork
from src.tune import tune
from src.classification_metrics import ClassificationMetrics
from src.utils import plot_combined_metrics
import os
import argparse
from src.neural_network import NeuralNetwork


class CommandLineInterface:

    def __init__(self):
        self.parser = self._get_parser()
        self.args = self._parse_args()
        self.data = Data(self.args.csv_path, drop_columns=["id"])
        self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(self.data.x, self.data.y, 0.2)

    def run(self):
        option = self._get_chosen_option()
        model = self._options()[option]()
        self._evaluate_model(model)
        if self.args.save:
            model.save(self.args.save)

    def _evaluate_model(self, model: NeuralNetwork):
        y_pred = model.predict(self.data.x)

        performance = ClassificationMetrics(y_pred, self.data.y)
        cm = performance.confusion_matrix()
        print("Precision:", performance.precision())
        print("Recall:", performance.recall())
        print("F1:", performance.f1())

        plot_combined_metrics(
            self.data.y, y_pred, cm, model.loss_history, model.accuracy_history, ["Negative", "Positive"]
        )

    def _from_pretrained(self):
        if not os.path.isdir(self.args.from_pretrained):
            raise FileNotFoundError(f"Trained model directory not found: {self.args.from_pretrained}")
        model = NeuralNetwork(from_pretrained=self.args.from_pretrained)
        print(f"Model loaded from '{self.args.from_pretrained}'")
        return model

    def _tuned_params(self):
        best_params, _ = tune(x_train=self.x_train, y_train=self.y_train, x_test=self.x_test, y_test=self.y_test)
        model = NeuralNetwork(n_inputs=self.x_train.shape[1], n_outputs=self.y_train.shape[1], **best_params)
        model.fit(self.x_train, self.y_train, show_training_output=True)
        print("Model trained with optimized hyperparameters")
        return model

    def _custom_params(self):
        if not (
            self.args.neurons
            and self.args.layers
            and self.args.outputs
            and self.args.learning_rate
            and self.args.epochs
        ):
            self.parser.error(
                "--neurons, --layers, --outputs, --learning-rate, and --epochs must be specified with --custom-parameters"
            )
        model = NeuralNetwork(
            n_inputs=self.x_train.shape[1],
            n_outputs=self.args.outputs,
            n_neurons=self.args.neurons,
            n_layers=self.args.layers,
            learning_rate=self.args.learning_rate,
            n_epochs=self.args.epochs,
        )
        model.fit(self.x_train, self.y_train, show_training_output=True)
        print(
            f"Model trained with user-defined parameters:\n - Neurons: {self.args.neurons}\n - Layers: {self.args.layers}\n - Outputs: {self.args.outputs}\n - Learning Rate: {self.args.learning_rate}\n - Epochs: {self.args.epochs}"
        )
        return model

    def _get_parser(self) -> argparse.ArgumentParser:
        return argparse.ArgumentParser(
            prog="multilayer-perceptron",
            description="This package allows easy customization, training, and performance evaluation of lightweight classification models.",
            epilog="Author: Arthur Bied-Charreton",
            formatter_class=argparse.RawTextHelpFormatter,
        )

    def _get_chosen_option(self):
        for option in ["from_pretrained", "tuned_parameters", "custom_parameters"]:
            if getattr(self.args, option.replace("-", "_")):
                return option
        raise ValueError("No valid option chosen")

    def _parse_args(self):
        group = self.parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-p", "--from-pretrained", metavar="<path>", help="path to your trained model", action="store", type=str
        )
        group.add_argument("-t", "--tuned-parameters", help="tune and train a new model", action="store_true")
        group.add_argument("-c", "--custom-parameters", help="enter custom training parameters", action="store_true")

        self.parser.add_argument("csv_path", metavar="<csv-path>", help="path to input data", type=str)

        self.parser.add_argument(
            "-n", "--neurons", metavar="<int>", help="neurons count for hidden layers", action="store", type=int
        )
        self.parser.add_argument(
            "-l", "--layers", metavar="<int>", help="number of dense layers", action="store", type=int
        )
        self.parser.add_argument(
            "-o",
            "--outputs",
            metavar="<int>",
            help="number of neurons in the output layer. use '--outputs 1' for binary classification",
            action="store",
            type=int,
        )
        self.parser.add_argument(
            "-r",
            "--learning-rate",
            metavar="<float>",
            help="learning rate for model training",
            action="store",
            type=float,
        )
        self.parser.add_argument(
            "-e", "--epochs", metavar="<int>", help="number of training iterations", action="store", type=int
        )

        self.parser.add_argument(
            "-s",
            "--save",
            metavar="<path>",
            help="path to which you want to save the model.\nWARNING: use a non-existing directory, as this will clear any existing content",
        )

        return self.parser.parse_args()

    def _validate_args(self):
        if (self.args.from_pretrained or self.args.tuned_parameters) and (
            self.args.neurons or self.args.layers or self.args.outputs or self.args.learning_rate or self.args.epochs
        ):
            self.parser.error("--neurons, --layers, --outputs and --epochs are only valid with --custom-params")

    def _validate_input_file(self, filename: str):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Input data file not found: {filename}")
        _, ext = os.path.splitext(filename)
        if ext.lower() != ".csv":
            raise ValueError(f"Invalid extension, expected: '.csv', got: '{ext}'")

    def _options(self):
        return {
            "from_pretrained": self._from_pretrained,
            "tuned_parameters": self._tuned_params,
            "custom_parameters": self._custom_params,
        }
