import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from sklearn.metrics import ConfusionMatrixDisplay

from visual_class_tuner.classes import ClassifierSettings, MockClassifier

# %% -
settings = ClassifierSettings(
    precision=0.95,
    recall=0.95,
    specificity=0.95,
    prob_distn=beta(0.25, 0.25),
    default_threshold=0.75,
)
classifier = MockClassifier.from_metrics(settings=settings)

# %% -
p = classifier.violin_view()
p.save("temp.pdf")

# %% -
display = classifier.plot_confusion_matrix()
display.plot()
plt.show()
