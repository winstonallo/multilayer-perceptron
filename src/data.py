import numpy as np
import pandas as pd
from numpy import ndarray

class Data:
    """
    A class to load and preprocess the dataset.
    """

    def __init__(self, csv_path: str, drop_columns: list[str] = None) -> None:
        self.df = self._load_csv(csv_path)
        self._preprocess(drop_columns)

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV file and handle potential errors.
        """
        try:
            df = pd.read_csv(csv_path, header=None)
            print("CSV file loaded successfully.")
            return df
        except FileNotFoundError:
            print(f"File not found: {csv_path}")
            raise
        except pd.errors.EmptyDataError:
            print("No data: CSV file is empty.")
            raise
        except pd.errors.ParserError:
            print("Error parsing CSV file.")
            raise

    def _preprocess(self, drop_columns: list[str]) -> None:
        """
        Preprocess the dataset by converting categorical data into numerical,
        and scaling the features and splitting the dataset into input and target.
        """
        self._add_columns()

        if drop_columns:
            self.df.drop(drop_columns, axis=1, inplace=True)

        self.df["diagnosis"] = self.df["diagnosis"].map({"M": 1, "B": 0})

        self.df = self.df.fillna(self.df.mean())

        self._handle_missing_values()
        self._scale_features()

        self.x = self.df.drop("diagnosis", axis=1).to_numpy()
        self.y = self.df["diagnosis"].to_numpy().reshape(-1, 1)

    def _handle_missing_values(self) -> None:
        """
        Impute missing values using a more sophisticated approach.
        """
        for column in self.df.columns:
            if self.df[column].isnull().any():
                self.df[column].fillna(self.df[column].mean(), inplace=True)
        print("Missing values imputed.")

    def _scale_features(self) -> None:
        """
        Standardize the features for faster convergence.
        """
        for column in self.df.columns:
            if column != "diagnosis":
                std = self.df[column].std()
                if std != 0:
                    self.df[column] = (self.df[column] - self.df[column].mean()) / std
                else:
                    self.df[column] = 0
        print("Features scaled.")

    def _add_columns(self) -> None:
        """
        Temporarily add column names to the dataset for easier preprocessing.
        """
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
        print("Columns added.")

def train_test_split(x: ndarray, y: ndarray, split: float) -> tuple:
    """
    Split the dataset randomly into training and testing sets with stratification,
    meaning the proportion of positive and negative results stays the same.
    """
    assert 0 < split < 1, "Split ratio must be between 0 and 1."

    unique_classes, class_counts = np.unique(y, return_counts=True)
    train_indices = []
    test_indices = []

    for cls, count in zip(unique_classes, class_counts):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        split_idx = int(count * (1 - split))
        train_indices.extend(cls_indices[:split_idx])
        test_indices.extend(cls_indices[split_idx:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    print("Data split into train and test sets.")

    return x_train, y_train, x_test, y_test
