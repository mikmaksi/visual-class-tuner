from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotnine as pn
from pydantic import Field, field_serializer, field_validator
from scipy.stats import beta
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve, roc_curve

from visual_class_tuner import rng
from visual_class_tuner.pydantic_config import Model

CLASS_NAME_TO_ACTUAL = {"TP": 1, "FN": 1, "FP": 0, "TN": 0}
CLASS_NAME_TO_PREDICTED = {"TP": 1, "FN": 0, "FP": 1, "TN": 0}


class ClassifierSettings(Model):
    precision: float  # TP/(TP+FP)
    recall: float  # TP/(TP+FN)
    specificity: float  # TN/(TN+FP)
    N: int = 1000  # number of samples
    threshold: float = 0.5  # default seting for the classifier for prediction

    # probability generation from TP, FN, FP, TN counts

    # distribution for sampling predicted probability values
    prob_distn: rv_continuous_frozen = beta(a=0.4, b=0.4)
    # threshold to split prob_distn by for additional control over distribution sampling
    generation_threshold: float = 0.5


class MockClassifier(Model):
    """MockClassifier.

    Args:
        y_true (np.ndarray): true class labels {0, 1} for each sample.
        y_prob (np.ndarray): probabilities returned by the classifier for each sample.
        threshold (float): threshold for converting probabilities into predicted labels.

    y_ture, y_prob and threshold uniquely define the mock classifier. To generate from performance metrics,
    use @cls.from_metrics
    """

    y_true: np.ndarray[int] = Field(repr=False)
    y_prob: np.ndarray[float] = Field(repr=False)
    threshold: float

    @field_serializer("y_true", "y_prob")
    def serialize_np_array(self, np_array: np.ndarray):
        return np_array.tolist()

    @field_validator("y_true", "y_prob", mode="before")
    @classmethod
    def convert_to_np_array(cls, value: list) -> np.ndarray:
        if isinstance(value, list):
            value = np.array(value)
        return value

    @classmethod
    def from_metrics(cls, settings: ClassifierSettings) -> "MockClassifier":
        """Instantiate from a set of classification performance metrics

        Args:
            settings (ClassifierSettings): a settings dataclass

        Returns:
            A mock classifier.
        """
        class_counts = cls.calculate_class_counts(settings.precision, settings.recall, settings.specificity, settings.N)
        y_true, y_prob = cls.make_samples(class_counts, settings.prob_distn, settings.generation_threshold)
        classifier = cls(y_true=y_true, y_prob=y_prob, threshold=settings.threshold)
        return classifier

    @staticmethod
    def calculate_class_counts(p: float, r: float, s: float, N: int) -> dict[str, int]:
        """Solve a system of linear equations for TP, FP, TN and FN

        Args:
            p (float): precision
            r (float): recall
            s (float): specificity
            N (int): number of samples

        Returns:
            dict[str, int]: dictionary of class labels and associated counts
        """
        X = np.array([[1, 1, 1, 1], [p - 1, p, 0, 0], [r - 1, 0, 0, r], [0, s, s - 1, 0]])
        Y = np.array([N, 0, 0, 0])
        class_counts = np.linalg.solve(X, Y)
        class_counts = class_counts.astype(int)
        return dict(zip(["TP", "FP", "TN", "FN"], class_counts))

    @staticmethod
    def make_samples(
        class_counts: dict[str, int],
        prob_distn: rv_continuous_frozen,
        generation_threshold: float,
    ) -> (np.ndarray[int], np.ndarray[int], np.ndarray[float]):
        """Generate samples corresponding to the desired class counts.

        Args:
            class_counts (dict[str, int]): dictionary of class labels and associated counts
            prob_distn (rv_continuous_frozen): probability distribution to sample probabilities from
            threshold (float): probability threshold separating the positive and negative classes

        Returns:
            (np.ndarray[int], np.ndarray[int], np.ndarray[float]):
        """

        # sample values from the probability distribution
        prob_samples = prob_distn.rvs(sum(class_counts.values()) * 3)
        pos_prob_samples = prob_samples[prob_samples >= generation_threshold]
        neg_prob_samples = prob_samples[prob_samples < generation_threshold]

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
        """Return predicted labels based on current threshold.

        Args:
            None

        Returns:
            np.ndarray[int]:
        """
        return (self.y_prob >= self.threshold).astype(int)

    @property
    def class_names(self) -> np.ndarray[str]:
        """Assign class names based on y_true and y_pred labels

        Args:
            None

        Returns:
            np.ndarray[str]:
        """
        class_names = np.empty(len(self.y_true), dtype="U2")
        matches = self.y_true == self.y_pred
        is_positive = np.array(self.y_true) == 1
        class_names[matches & is_positive] = "TP"
        class_names[~matches & is_positive] = "FN"
        class_names[~matches & ~is_positive] = "FP"
        class_names[matches & ~is_positive] = "TN"
        return class_names

    def to_df(self) -> pd.DataFrame:
        """Create a DataFrame summary by sample.

        Args:
            None

        Returns:
            pd.DataFrame:
        """
        samples_df = np.array([self.class_names, self.y_true, self.y_pred, self.y_prob]).T
        samples_df = pd.DataFrame(samples_df, columns=["class", "actual", "predicted", "prob"]).astype({"prob": float})

        # set categories
        samples_df["class"] = pd.Categorical(samples_df["class"], categories=np.unique(self.class_names))
        for label_type in ["actual", "predicted"]:
            samples_df[label_type] = pd.Categorical(samples_df[label_type], categories=["1", "0"])

        return samples_df

    @property
    def TP(self) -> int:
        return (self.class_names == "TP").sum()

    @property
    def FP(self) -> int:
        return (self.class_names == "FP").sum()

    @property
    def TN(self) -> int:
        return (self.class_names == "TN").sum()

    @property
    def FN(self) -> int:
        return (self.class_names == "FN").sum()

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
    def recall(self):
        return self.TP / self.P

    @property
    def fnr(self):
        return self.FN / self.P

    @property
    def fpr(self):
        return self.FP / self.N

    @property
    def specificity(self):
        return self.TN / self.N

    @property
    def precision(self):
        return self.TP / self.PP

    @property
    def fomr(self):
        return self.FN / self.PN

    @property
    def fdr(self):
        return self.FP / self.PP

    @property
    def npv(self):
        return self.TN / self.PN

    def plot_confusion_matrix(self, engine: Literal["plotly", "matplotlib"] = "plotly"):
        if engine == "plotly":
            p = px.imshow(
                img=self.confusion_matrix.T,
                x=["1", "0"],
                y=["1", "0"],
                text_auto=True,
                labels={"x": "True label", "y": "Predicted label"},
            )
            p.update_xaxes(side="top")
            p = p.update_coloraxes(showscale=False)
            p.write_html("temp.html")
            return p
        elif engine == "matplotlib":
            display = ConfusionMatrixDisplay(self.confusion_matrix)
            _ = display.plot()
            return display.figure_

    def plot_violins(self, engine: Literal["plotly", "plotnine"] = "plotly"):
        """Plot distribution of probabilities as a violing plot"""
        if engine == "plotly":
            p = px.violin(data_frame=self.to_df(), x="actual", y="prob", color="class", violinmode="overlay")
            strip = px.strip(data_frame=self.to_df(), x="actual", y="prob", color="class")
            p.add_traces(list(strip.select_traces()))
            p.add_hline(y=self.threshold, line_dash="dash")
            p.write_html("temp.html")
        elif engine == "plotnine":
            p = pn.ggplot(self.to_df(), pn.aes("actual", "prob"))
            p = p + pn.geom_violin()
            p = p + pn.geom_sina(mapping=pn.aes(color="class"), position=pn.position_dodge(width=0))
            p = p + pn.geom_hline(yintercept=self.threshold, linetype="dashed")
        return p

    def plot_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob, drop_intermediate=True)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
        fig = px.line(data_frame=roc_df, x="fpr", y="tpr", markers=True, hover_data={"threshold": ":0.3f"})
        thresh_idx = np.argmin(np.abs(thresholds - self.threshold))
        fig = fig.add_hline(y=tpr[thresh_idx], line_dash="dash")
        fig = fig.add_vline(x=fpr[thresh_idx], line_dash="dash")
        return fig

    def plot_precision_recall_curve(self):
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_prob, drop_intermediate=True)
        pr_df = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds})
        fig = px.line(data_frame=pr_df, x="recall", y="precision", markers=True, hover_data={"threshold": ":0.3f"})
        thresh_idx = np.argmin(np.abs(thresholds - self.threshold))
        fig = fig.add_hline(y=precision[thresh_idx], line_dash="dash")
        fig = fig.add_vline(x=recall[thresh_idx], line_dash="dash")
        return fig
