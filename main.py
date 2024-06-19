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
    epilog="Author: Arthur Bied-Charreton"
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--from-pretrained", metavar="<path to saved model>", help="path to your trained model", action="store")
group.add_argument("--optimized-training", help="tune and train a new model", action="store_true")
group.add_argument("--custom-params", help="enter custom training parameters", action="store_true")

parser.add_argument("data.csv", help="path to input data")

parser.add_argument("--neurons", metavar="<number of neurons>", help="neurons count for hidden layers", action="store")
parser.add_argument("--layers", metavar="<number of layers>", help="number of dense layers", action="store")


args = parser.parse_args()

if (args.from_pretrained or args.optimized_training) and (args.neurons or args.layers):
    parser.error("--neurons and --layers are only valid with --custom-params")

print(args)
