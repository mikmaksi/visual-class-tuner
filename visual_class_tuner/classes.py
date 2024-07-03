from class_visualizer.pydantic_config import Model
import pandas as pd
import numpy as np
import plotnine as pn
from pydantic import model_validator
from typing import ClassVar
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from scipy.stats import beta
from class_visualizer import rng


class ClassifierSettings(Model):
    precision: float  # TP/(TP+FP)
    recall: float  # TP/(TP+FN)
    specificity: float  # TN/(TN+FP)
    N: int = 1000
    prob_distn: rv_continuous_frozen = beta(a=0.4, b=0.4)
    default_threshold: float = 0.5


class MockClassifier(Model):
    settings: ClassifierSettings
    TP: int = None
    FP: int = None
    TN: int = None
    FN: int = None
    PP: int = None
    PN: int = None
    P: int = None
    N: int = None
    acc: float = None
    samples: pd.DataFrame = None
    class_names: ClassVar[list[str]] = ["TP", "FP", "TN", "FN"]

    @model_validator(mode="after")
    def solve(self) -> "MockClassifier":
        """Solve a system of linear equations for TP, FP, TN and FN"""
        # extract params from settings
        p = self.settings.precision
        r = self.settings.recall
        s = self.settings.specificity
        N = self.settings.N

        # solve system of equations
        A = np.array([[1, 1, 1, 1], [p - 1, p, 0, 0], [r - 1, 0, 0, r], [0, s, s - 1, 0]])
        B = np.array([N, 0, 0, 0])
        class_counts = np.linalg.solve(A, B)

        # update attributes
        for class_name, n in zip(self.class_names, class_counts):
            setattr(self, class_name, int(n))
        self.acc = (self.TP + self.TN) / self.settings.N
        self.PP = self.TP + self.FP
        self.PN = self.FN + self.TN
        self.P = self.TP + self.FN
        self.N = self.FP + self.TN
        return self

    @property
    def confusion_matrix(self):
        return np.array([[self.TP, self.FN], [self.FP, self.TN]])

    @property
    def npv(self):
        return self.TN / self.PN

    def set_samples(self) -> pd.DataFrame:
        """Generate samples corresponding to the desired class counts"""

        # sample values from the probability distribution
        prob_samples = self.settings.prob_distn.rvs(self.settings.N * 3)
        pos_prob_samples = prob_samples[prob_samples >= self.settings.default_threshold]
        neg_prob_samples = prob_samples[prob_samples < self.settings.default_threshold]

        # generate a DataFrame of class labels and probabilities
        class_names = []
        actual_label = []
        pred_label = []
        probs = []
        for (actual, predicted), class_name in zip([(1, 1), (1, 0), (0, 0), (0, 1)], self.class_names):
            n = getattr(self, class_name)
            class_names.extend([class_name] * n)
            actual_label.extend([actual] * n)
            pred_label.extend([predicted] * n)
            if predicted == 1:
                probs.extend(rng.choice(pos_prob_samples, n))
            else:
                probs.extend(rng.choice(neg_prob_samples, n))
        samples = np.array([class_names, actual_label, pred_label, probs]).T
        samples = pd.DataFrame(samples, columns=["class", "actual", "predicted", "prob"]).astype({"prob": float})

        # set categories
        samples["class"] = pd.Categorical(samples["class"], categories=self.class_names)
        for label_type in ["actual", "predicted"]:
            samples[label_type] = pd.Categorical(samples[label_type], categories=["1", "0"])

        self.samples = samples

    def plot_confusion_matrix(self):
        display = ConfusionMatrixDisplay(self.confusion_matrix)
        return display

    def violin_view(self):
        """Plot distribution of probabilities as a violing plot"""
        p = pn.ggplot(self.samples, pn.aes("actual", "prob"))
        p = p + pn.geom_violin()
        p = p + pn.geom_sina(mapping=pn.aes(color="class"), position=pn.position_dodge(width=0))
        p = p + pn.geom_hline(yintercept=self.settings.default_threshold, linetype="dashed")
        return p
