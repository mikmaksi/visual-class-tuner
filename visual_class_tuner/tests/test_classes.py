import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import beta, rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve

from visual_class_tuner.classes import ClassifierSettings, MockClassifier

# %% -
settings = ClassifierSettings(
    precision=0.95,
    recall=0.95,
    specificity=0.95,
    threshold=0.5,
    prob_distn=beta(0.25, 0.25),
)
classifier = MockClassifier.from_metrics(settings=settings)

# %% -
classifier.threshold = 0.8
p = classifier.plot_violins()
# p.save("temp.pdf")
p.write_html("temp.html")

# %% -
fig = classifier.plot_confusion_matrix()
fig.write_html("temp.html")

# %% -
fig = classifier.plot_roc_curve()
fig.write_html("temp.html")

# %% -
fig = classifier.plot_precision_recall_curve()
fig.write_html("temp.html")

# %% -
# (de)serialization
MockClassifier(**classifier.model_dump())
