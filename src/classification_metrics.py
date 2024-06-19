import numpy as np
from numpy import ndarray


class ClassificationMetrics:
    def __init__(self, y_pred: ndarray, y_true: ndarray):
        self.y_pred = y_pred
        self.y_true = y_true
        self._preprocess()

    def _preprocess(self):
        y_pred_rounded = np.round(self.y_pred).astype(int)

        y_true_int = self.y_true.astype(int)

        self.true_pos = np.sum(np.logical_and(y_true_int == 1, y_pred_rounded == 1))
        self.false_pos = np.sum(np.logical_and(y_true_int == 0, y_pred_rounded == 1))
        self.true_neg = np.sum(np.logical_and(y_true_int == 0, y_pred_rounded == 0))
        self.false_neg = np.sum(np.logical_and(y_true_int == 1, y_pred_rounded == 0))

        self.total_error_count = self.false_neg + self.false_pos

    def precision(self) -> float:
        denominator = self.true_pos + self.false_pos
        if denominator == 0:
            return 0.0
        return self.true_pos / denominator

    def recall(self) -> float:
        denominator = self.true_pos + self.false_neg
        if denominator == 0:
            return 0.0
        return self.true_pos / denominator

    def f1(self) -> float:
        prec = self.precision()
        rec = self.recall()
        denominator = prec + rec
        if denominator == 0:
            return 0.0
        return 2 * (prec * rec) / denominator

    def confusion_matrix(self) -> np.ndarray:
        return np.array([
            [self.true_neg, self.false_pos],
            [self.false_neg, self.true_pos]
        ])
