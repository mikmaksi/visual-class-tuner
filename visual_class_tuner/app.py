import re

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback
from dash import callback_context as ctx
from dash import dcc, html
from scipy import stats
from scipy.stats import beta
from scipy.stats._continuous_distns import _distn_names as cont_distns
from scipy.stats._discrete_distns import _distn_names as disc_distns

from visual_class_tuner.classes import ClassifierSettings, MockClassifier

# random number generator
rng = np.random.default_rng()

# create the app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css", "assets/styles.css"]
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

confusion_matrix_table = dbc.Row(
    [
        dbc.Row(
            [
                dbc.Col("Total population"),
                dbc.Col("Predicted Positive (PP)"),
                dbc.Col("Predicted Positive (PN)"),
                dbc.Col(),
                dbc.Col(),
            ],
            className="cm-table header",
        ),
        dbc.Row(
            [
                dbc.Col("Positive (P)", className="cm-table header"),
                dbc.Col(id="tbl-tp"),
                dbc.Col(id="tbl-fn"),
                dbc.Col(id="tbl-recall"),
                dbc.Col(id="tbl-fnr"),
            ]
        ),
        dbc.Row(
            [
                dbc.Col("Negative (N)", className="cm-table header"),
                dbc.Col(id="tbl-fp"),
                dbc.Col(id="tbl-tn"),
                dbc.Col(id="tbl-fpr"),
                dbc.Col(id="tbl-specificity"),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(className="cm-table header"),
                dbc.Col(id="tbl-precision"),
                dbc.Col(id="tbl-for"),
                dbc.Col(),
                dbc.Col(),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(className="cm-table header"),
                dbc.Col(id="tbl-fdr"),
                dbc.Col(id="tbl-npv"),
                dbc.Col(),
                dbc.Col(),
            ]
        ),
    ],
    className="cm-table",
)


# build the layout
app.layout = dbc.Container(
    [
        html.H1("Build the classifier"),
        dbc.Stack(
            children=[
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Precision", class_name="text"),
                        dbc.Input(id="precision-input", value=0.9, type="number", min=0, max=1, step=0.05),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Recall", class_name="text"),
                        dbc.Input(id="recall-input", value=0.9, type="number", min=0, max=1, step=0.05),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Specificity", class_name="text"),
                        dbc.Input(id="specificity-input", value=0.9, type="number", min=0, max=1, step=0.05),
                    ]
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("N", class_name="text"),
                        dbc.Input(id="n-samples-input", value=500, type="number", min=50, max=10000, step=10),
                    ]
                ),
            ], class_name="params-input-group"
        ),
        html.Hr(),
        html.H1("Adjust Threshold"),
        dbc.Row(
            [
                html.Label("Threshold"),
                dcc.Slider(id="threshold-input", value=0.5, min=0, max=1, step=0.05),
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(confusion_matrix_table, style={"align-content": "center"}, width=6),
                dbc.Col(dcc.Graph(id="violins-plot"), width=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="roc-curve"), width=6),
                dbc.Col(dcc.Graph(id="precision-recall-curve"), width=6),
            ],
        ),
        # storage
        dcc.Store(id="classifier", storage_type="memory"),
    ],
    fluid=True,
)


@callback(
    Output("classifier", "data"),
    Input("precision-input", "value"),
    Input("recall-input", "value"),
    Input("specificity-input", "value"),
    Input("n-samples-input", "value"),
    State("threshold-input", "value"),
)
def make_classifier(precision, recall, specificity, n_samples, threshold):
    settings = ClassifierSettings(
        precision=precision,
        recall=recall,
        specificity=specificity,
        N=n_samples,
        prob_distn=beta(0.25, 0.25),
        threshold=threshold,
    )
    classifier = MockClassifier.from_metrics(settings=settings)
    return classifier.model_dump()


@callback(
    Output("classifier", "data", allow_duplicate=True),
    Input("threshold-input", "value"),
    State("classifier", "data"),
    prevent_initial_call=True,
)
def update_classifier(threshold: float, classifier_dict: dict):
    classifier = MockClassifier(**classifier_dict)
    classifier.threshold = threshold
    return classifier.model_dump()


@callback(
    Output("tbl-tp", "children"),
    Output("tbl-fp", "children"),
    Output("tbl-fn", "children"),
    Output("tbl-tn", "children"),
    Output("tbl-recall", "children"),
    Output("tbl-fnr", "children"),
    Output("tbl-fpr", "children"),
    Output("tbl-specificity", "children"),
    Output("tbl-precision", "children"),
    Output("tbl-for", "children"),
    Output("tbl-fdr", "children"),
    Output("tbl-npv", "children"),
    Input("classifier", "data"),
)
def update_table(classifier_dict: dict):
    classifier = MockClassifier(**classifier_dict)
    output = (
        f"TP = {classifier.TP}",
        f"FP = {classifier.FP}",
        f"FN = {classifier.FN}",
        f"TN = {classifier.TN}",
        f"Recall/Sensitivity/TPR = {classifier.recall:.3f}",
        f"FNR = {classifier.fnr:.3f}",
        f"FPR = {classifier.fpr:.3f}",
        f"Specificity/TNR = {classifier.specificity:.3f}",
        f"Precision/PPV = {classifier.precision:.3f}",
        f"FOR = {classifier.fomr:.3f}",
        f"FDR = {classifier.fdr:.3f}",
        f"NPV = {classifier.npv:.3f}",
    )
    return output


@callback(Output("violins-plot", "figure"), Input("classifier", "data"))
def update_violins(classifier_dict: dict):
    classifier = MockClassifier(**classifier_dict)
    return classifier.plot_violins(engine="plotly")


@callback(Output("roc-curve", "figure"), Input("classifier", "data"))
def update_roc_curve(classifier_dict: dict):
    classifier = MockClassifier(**classifier_dict)
    return classifier.plot_roc_curve()


@callback(Output("precision-recall-curve", "figure"), Input("classifier", "data"))
def update_precision_recall_curve(classifier_dict: dict):
    classifier = MockClassifier(**classifier_dict)
    return classifier.plot_precision_recall_curve()


if __name__ == "__main__":
    app.run(debug=True)
