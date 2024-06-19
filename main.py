from src.data import Data, train_test_split
from src.neural_network import NeuralNetwork
from src.tune import tune
from src.classification_metrics import ClassificationMetrics
from src.utils import plot_combined_metrics
from simple_term_menu import TerminalMenu
import sys
import os
import argparse

def get_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="multilayer-perceptron",
        description="This package allows easy customization, training, and performance evaluation of lightweight classification models.",
        epilog="Author: Arthur Bied-Charreton",
        formatter_class=argparse.RawTextHelpFormatter,
    )

def parse_args(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p", "--from-pretrained", metavar="<path>", help="path to your trained model", action="store", type=str
    )
    group.add_argument("-t", "--tuned-parameters", help="tune and train a new model", action="store_true")
    group.add_argument("-c", "--custom-parameters", help="enter custom training parameters", action="store_true")

    parser.add_argument("csv_path", metavar="<csv-path>", help="path to input data", type=str)

    parser.add_argument(
        "-n", "--neurons", metavar="<int>", help="neurons count for hidden layers", action="store", type=int
    )
    parser.add_argument("-l", "--layers", metavar="<int>", help="number of dense layers", action="store", type=int)
    parser.add_argument(
        "-o",
        "--outputs",
        metavar="<int>",
        help="number of neurons in the output layer. use '--outputs 1' for binary classification",
        action="store",
        type=int,
    )
    parser.add_argument(
        "-r", "--learning-rate", metavar="<float>", help="learning rate for model training", action="store", type=float
    )
    parser.add_argument("-e", "--epochs", metavar="<int>", help="number of training iterations", action="store", type=int)

    parser.add_argument(
        "-s",
        "--save",
        metavar="<path>",
        help="path to which you want to save the model.\nWARNING: use a non-existing directory, as this will clear any existing content",
    )

    return parser.parse_args()

def validate_args(args, parser):
    if (args.from_pretrained or args.tuned_parameters) and (
        args.neurons or args.layers or args.outputs or args.learning_rate or args.epochs
    ):
        parser.error("--neurons, --layers, --outputs and --epochs are only valid with --custom-params")

def validate_input_file(data_path: str):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Input data file not found: {data_path}")

def main():
    parser = get_parser()
    args = parse_args(parser)
    validate_args(args, parser)
    data_path = args.csv_path
    try:
        validate_input_file(data_path)
        data = Data(data_path, drop_columns=["id"])
        x_train, y_train, x_test, y_test = train_test_split(data.x, data.y, split=0.2)
        if args.from_pretrained:
            if not os.path.isdir(args.from_pretrained):
                raise FileNotFoundError(f"Trained model directory not found: {args.from_pretrained}")
            model = NeuralNetwork(from_pretrained=args.from_pretrained)
            print(f"Model loaded from '{args.from_pretrained}'")
        elif args.tuned_parameters:
            best_params, results = tune(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            model = NeuralNetwork(n_inputs=x_train.shape[1], n_outputs=y_train.shape[1], **best_params)
            model.fit(x_train, y_train)
            print("Model trained")
        elif args.custom_parameters:
            if not (args.neurons and args.layers and args.outputs and args.learning_rate and args.epochs):
                parser.error(
                    "--neurons, --layers, --outputs, --learning-rate, and --epochs must be specified with --custom-parameters"
                )
            model = NeuralNetwork(
                n_inputs=x_train.shape[1], 
                n_outputs=args.outputs, 
                n_neurons=args.neurons, 
                n_layers=args.layers, 
                learning_rate=args.learning_rate, 
                n_epochs=args.epochs
            )
            model.fit(x_train, y_train)
            print("Model trained")
        else:
            raise ValueError("Invalid argument combination")
        y_pred = model.predict(data.x)
        performance = ClassificationMetrics(y_pred, data.y)
        cm = performance.confusion_matrix()
        print("Confusion matrix:\n", cm)

        print("Precision:", performance.precision())
        print("Recall:", performance.recall())
        print("F1:", performance.f1())

        # plot_combined_metrics(data.y, y_pred, cm, model.loss_history, model.accuracy_history, ["Negative", "Positive"])
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()
