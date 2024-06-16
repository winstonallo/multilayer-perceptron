import pandas as pd
import numpy as np
from numpy import ndarray


class Input:

    def __init__(self):
        self.load_data()

    def get_data(self) -> tuple[ndarray, ndarray]:
        self.inputs = self.inputs.to_numpy()
        self.targets = self.targets.to_numpy()
        return self.inputs, self.targets


class TrainingData(Input):

    def load_data(self) -> None:
        self.inputs = pd.read_csv("data/train/inputs.csv")
        self.targets = pd.read_csv("data/train/targets.csv")


class TestData(Input):

    def load_data(self) -> None:
        self.inputs = pd.read_csv("data/test/inputs.csv")
        self.targets = pd.read_csv("data/test/targets.csv")
