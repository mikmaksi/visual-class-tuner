from typing import ClassVar

import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import beta
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from sklearn.metrics import ConfusionMatrixDisplay
from pydantic import Field

from visual_class_tuner import rng
from visual_class_tuner.pydantic_config import Model


class ClassifierSettings(Model):
    precision: float  # TP/(TP+FP)
    recall: float  # TP/(TP+FN)
    specificity: float  # TN/(TN+FP)
    N: int = 1000
    prob_distn: rv_continuous_frozen = beta(a=0.4, b=0.4)
    default_threshold: float = 0.5


class MockClassifier(Model):
    class_names: ClassVar[list[str]] = ["TP", "FP", "TN", "FN"]
    class_labels: ClassVar[list[tuple[int, int]]] = [(1, 1), (1, 0), (0, 0), (0, 1)]

    y_true: np.ndarray = Field(repr=False)
    y_pred: np.ndarray = Field(repr=False)
    y_prob: np.ndarray = Field(repr=False)
    threshold: float

    @classmethod
    def from_metrics(cls, settings: ClassifierSettings):
        class_counts = cls.calculate_class_counts(settings.precision, settings.recall, settings.specificity, settings.N)
        y_true, y_pred, y_prob = cls.make_samples(
            class_counts, cls.class_labels, settings.prob_distn, settings.default_threshold
        )
        return cls(y_true=y_true, y_pred=y_pred, y_prob=y_prob, threshold=settings.default_threshold)

    @staticmethod
    def calculate_class_counts(p: float, r: float, s: float, N: int) -> np.ndarray:
        """Solve a system of linear equations for TP, FP, TN and FN"""
        # solve system of equations
        X = np.array([[1, 1, 1, 1], [p - 1, p, 0, 0], [r - 1, 0, 0, r], [0, s, s - 1, 0]])
        Y = np.array([N, 0, 0, 0])
        class_counts = np.linalg.solve(X, Y)
        class_counts = class_counts.astype(int)
        return class_counts

    @staticmethod
    def make_samples(
        class_counts: list[int], class_labels: list[tuple[int, int]], prob_distn: rv_continuous_frozen, threshold: float
    ) -> (np.ndarray[int], np.ndarray[int], np.ndarray[float]):
        """Generate samples corresponding to the desired class counts"""

        # sample values from the probability distribution
        prob_samples = prob_distn.rvs(class_counts.sum() * 3)
        pos_prob_samples = prob_samples[prob_samples >= threshold]
        neg_prob_samples = prob_samples[prob_samples < threshold]

        # generate a DataFrame of class labels and probabilities
        y_true = []
        y_pred = []
        y_prob = []
        for n, (actual, predicted) in zip(class_counts, class_labels):
            y_true.extend([actual] * n)
            y_pred.extend([predicted] * n)
            if predicted == 1:
                y_prob.extend(rng.choice(pos_prob_samples, n))
            else:
                y_prob.extend(rng.choice(neg_prob_samples, n))
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        return y_true, y_pred, y_prob

    @property
    def labels(self) -> np.ndarray[str]:
        labels = np.empty(len(self.y_true), dtype="U2")
        matches = np.array(self.y_true) == np.array(self.y_pred)
        is_positive = np.array(self.y_true) == 1
        labels[matches & is_positive] = "TP"
        labels[matches & ~is_positive] = "TN"
        labels[~matches & is_positive] = "FP"
        labels[~matches & ~is_positive] = "FN"
        return labels

    def to_df(self) -> pd.DataFrame:
        samples_df = np.array([self.labels, self.y_true, self.y_pred, self.y_prob]).T
        samples_df = pd.DataFrame(samples_df, columns=["class", "actual", "predicted", "prob"]).astype({"prob": float})

        # set categories
        samples_df["class"] = pd.Categorical(samples_df["class"], categories=self.class_names)
        for label_type in ["actual", "predicted"]:
            samples_df[label_type] = pd.Categorical(samples_df[label_type], categories=["1", "0"])

        return samples_df

    @property
    def TP(self) -> int:
        return (self.labels == "TP").sum()

    @property
    def FP(self) -> int:
        return (self.labels == "FP").sum()

    @property
    def TN(self) -> int:
        return (self.labels == "TN").sum()

    @property
    def FN(self) -> int:
        return (self.labels == "FN").sum()

    @property
    def PP(self) -> int:
        return self.TP + self.FP

    @property
    def PN(self) -> int:
        return self.FN + self.TN

    @property
    def P(self) -> int:
        return self.TP + self.FN

    @property
    def N(self) -> int:
        return self.FP + self.TN

    @property
    def confusion_matrix(self):
        return np.array([[self.TP, self.FN], [self.FP, self.TN]])

    @property
    def acc(self) -> int:
        return (self.TP + self.TN) / len(self.y_true)

    @property
    def npv(self):
        return self.TN / self.PN

    def plot_confusion_matrix(self):
        display = ConfusionMatrixDisplay(self.confusion_matrix)
        return display

    def violin_view(self):
        """Plot distribution of probabilities as a violing plot"""
        p = pn.ggplot(self.to_df(), pn.aes("actual", "prob"))
        p = p + pn.geom_violin()
        p = p + pn.geom_sina(mapping=pn.aes(color="class"), position=pn.position_dodge(width=0))
        p = p + pn.geom_hline(yintercept=self.threshold, linetype="dashed")
        return p
