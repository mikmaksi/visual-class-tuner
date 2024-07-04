from typing import ClassVar

import numpy as np
import pandas as pd
import plotnine as pn
from pydantic import Field
from scipy.stats import beta
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from sklearn.metrics import ConfusionMatrixDisplay

from visual_class_tuner import rng
from visual_class_tuner.pydantic_config import Model

CLASS_NAME_TO_ACTUAL = {"TP": 1, "FN": 1, "FP": 0, "TN": 0}
CLASS_NAME_TO_PREDICTED = {"TP": 1, "FN": 0, "FP": 1, "TN": 0}


class ClassifierSettings(Model):
    precision: float  # TP/(TP+FP)
    recall: float  # TP/(TP+FN)
    specificity: float  # TN/(TN+FP)
    N: int = 1000
    prob_distn: rv_continuous_frozen = beta(a=0.4, b=0.4)
    threshold: float = 0.5


class MockClassifier(Model):
    y_true: np.ndarray = Field(repr=False)
    y_prob: np.ndarray = Field(repr=False)
    threshold: float  # this can be reset to recalculate y_pred

    @classmethod
    def from_metrics(cls, settings: ClassifierSettings):
        class_counts = cls.calculate_class_counts(settings.precision, settings.recall, settings.specificity, settings.N)
        y_true, y_prob = cls.make_samples(class_counts, settings.prob_distn, settings.threshold)
        return cls(y_true=y_true, y_prob=y_prob, threshold=settings.threshold)

    @staticmethod
    def calculate_class_counts(p: float, r: float, s: float, N: int) -> dict[str, int]:
        """Solve a system of linear equations for TP, FP, TN and FN"""
        # solve system of equations
        X = np.array([[1, 1, 1, 1], [p - 1, p, 0, 0], [r - 1, 0, 0, r], [0, s, s - 1, 0]])
        Y = np.array([N, 0, 0, 0])
        class_counts = np.linalg.solve(X, Y)
        class_counts = class_counts.astype(int)
        return dict(zip(["TP", "FP", "TN", "FN"], class_counts))

    @staticmethod
    def make_samples(
        class_counts: dict[str, int],
        prob_distn: rv_continuous_frozen,
        threshold: float,
    ) -> (np.ndarray[int], np.ndarray[int], np.ndarray[float]):
        """Generate samples corresponding to the desired class counts"""

        # sample values from the probability distribution
        prob_samples = prob_distn.rvs(sum(class_counts.values()) * 3)
        pos_prob_samples = prob_samples[prob_samples >= threshold]
        neg_prob_samples = prob_samples[prob_samples < threshold]

        # generate a DataFrame of class labels and probabilities
        y_true = []
        y_prob = []
        for class_name, n in class_counts.items():
            y_true.extend([CLASS_NAME_TO_ACTUAL[class_name]] * n)
            if CLASS_NAME_TO_PREDICTED[class_name] == 1:
                y_prob.extend(rng.choice(pos_prob_samples, n).tolist())
            else:
                y_prob.extend(rng.choice(neg_prob_samples, n).tolist())
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        return y_true, y_prob

    @property
    def y_pred(self) -> np.ndarray[int]:
        return (self.y_prob >= self.threshold).astype(int)

    @property
    def labels(self) -> np.ndarray[str]:
        labels = np.empty(len(self.y_true), dtype="U2")
        matches = np.array(self.y_true) == np.array(self.y_pred)
        is_positive = np.array(self.y_true) == 1
        labels[matches & is_positive] = "TP"
        labels[~matches & is_positive] = "FN"
        labels[~matches & ~is_positive] = "FP"
        labels[matches & ~is_positive] = "TN"
        return labels

    def to_df(self) -> pd.DataFrame:
        samples_df = np.array([self.labels, self.y_true, self.y_pred, self.y_prob]).T
        samples_df = pd.DataFrame(samples_df, columns=["class", "actual", "predicted", "prob"]).astype({"prob": float})

        # set categories
        samples_df["class"] = pd.Categorical(samples_df["class"], categories=np.unique(self.labels))
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
