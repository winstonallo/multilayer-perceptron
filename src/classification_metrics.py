import numpy as np
from numpy import ndarray


class ClassificationMetrics:

    def __init__(self, y_true: ndarray, y_pred: ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
        self.true_pos = None
        self.true_neg = None
        self.false_pos = None
        self.false_neg = None
        self.total_error_count = None
        self._preprocess()

    def confusion_matrix(self) -> ndarray:
        return np.array([[self.true_pos, self.false_pos], [self.true_neg, self.false_neg]])

    def precision(self) -> float:
        return self.true_pos / (self.true_pos + self.false_pos)

    def recall(self) -> float:
        return self.true_pos / (self.true_pos + self.false_neg)

    def f1(self) -> float:
        return 2 * (self.precision() * self.recall() / (self.precision() + self.recall()))

    def total_errors(self) -> int:
        return self.total_error_count

    def _preprocess(self):
        y_pred_rounded = np.round(self.y_pred).astype(int)
        y_true_int = self.y_true.astype(int)

        self.true_pos = np.sum(np.logical_and(y_true_int == 1, y_pred_rounded == 1))
        self.false_pos = np.sum(np.logical_and(y_true_int == 0, y_pred_rounded == 1))
        self.true_neg = np.sum(np.logical_and(y_true_int == 0, y_pred_rounded == 0))
        self.false_neg = np.sum(np.logical_and(y_true_int == 1, y_pred_rounded == 0))

        self.total_error_count = self.false_neg + self.false_pos
