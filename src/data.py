"""
This module contains the Data class which is responsible for loading the dataset
and preprocessing it. The train_test_split function is used to split the dataset
into training and testing sets.
"""

import pandas as pd
import numpy as np
from numpy import ndarray


class Data:
    """
    A class to load and preprocess the dataset.
    """

    def __init__(self, csv_path: str, drop_columns: list[str] = None) -> None:
        # The data is missing headers, header=None prevents pandas from interpreting
        # the first row as column names
        self.df = pd.read_csv(csv_path, header=None)
        self._preprocess(drop_columns)

    def _preprocess(self, drop_columns: list[str]) -> None:
        """
        Preprocess the dataset by converting categorical data into numerical,
        and scaling the features and splitting the dataset into input and target.
        """

        self._add_columns()  # Temporarily add the columns for easier preprocessing

        if drop_columns:
            self.df.drop(drop_columns, axis=1, inplace=True)

        self.df["diagnosis"] = self.df["diagnosis"].map({"M": 1, "B": 0})

        self.df = self.df.fillna(self.df.mean())  # Fill missing values with the mean

        for column in self.df.columns:  # Standardize the features
            if column != "diagnosis":
                self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()

        self.x = self.df.drop("diagnosis", axis=1).to_numpy()
        self.y = self.df["diagnosis"].to_numpy().reshape(-1, 1)

    def _add_columns(self) -> None:
        columns = [
            "id",
            "diagnosis",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ]

        self.df.columns = columns


def train_test_split(x: ndarray, y: ndarray, split: float) -> tuple:
    """
    Split the dataset randomly into training and testing sets.
    """
    assert 0 < split < 1, "Split ratio must be between 0 and 1."

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x_shuffled = x[indices]
    y_shuffled = y[indices]

    split_idx = int(x.shape[0] * (1 - split))

    x_train = x_shuffled[:split_idx]
    y_train = y_shuffled[:split_idx]
    x_test = x_shuffled[split_idx:]
    y_test = y_shuffled[split_idx:]

    return x_train, y_train, x_test, y_test
