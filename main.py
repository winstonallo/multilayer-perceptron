from src.data import Data, train_test_split
from src.neural_network import NeuralNetwork
from src.tune import tune
from src.classification_metrics import ClassificationMetrics
from src.utils import plot_combined_metrics
from simple_term_menu import TerminalMenu
import sys
import os
import argparse

parser = argparse.ArgumentParser(
    prog="multilayer-perceptron",
    description="This package allows easy customization, training, and performance evaluation of lightweight classification models.",
    epilog="Author: Arthur Bied-Charreton",
    formatter_class=argparse.RawTextHelpFormatter,
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-p", "--from-pretrained", metavar="<path>", help="path to your trained model", action="store", type=str
)
group.add_argument("-t", "--tuned-parameters", help="tune and train a new model", action="store_true")
group.add_argument("-c", "--custom-parameters", help="enter custom training parameters", action="store_true")

parser.add_argument("csv-path", metavar="<csv-path>", help="path to input data", type=str)

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

args = parser.parse_args()

if (args.from_pretrained or args.optimized_training) and (
    args.neurons or args.layers or args.outputs or args.learning_rate or args.epochs
):
    parser.error("--neurons, --layers, --outputs and --epochs are only valid with --custom-params")

print(args)
