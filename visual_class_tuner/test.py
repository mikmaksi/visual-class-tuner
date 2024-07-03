from class_visualizer.classes import ClassifierSettings, MockClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen

# %% -
settings = ClassifierSettings(
    precision=0.95,
    recall=0.95,
    specificity=0.95,
    prob_distn=beta(0.25, 0.25),
    default_threshold=0.75,
)
classifier = MockClassifier(settings=settings)

classifier.set_samples()
classifier.samples
p = classifier.violin_view()
p.save("temp.pdf")

# %% -
display = classifier.plot_confusion_matrix()
display.plot()
plt.show()

