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
from visual_class_tuner import rng, FIG_MARGIN

# create the app
external_stylesheets = [dbc.themes.FLATLY, "assets/styles.css"]
app = Dash(external_stylesheets=external_stylesheets)

# confusion matrix table layout element
confusion_matrix_table = {
    "Total population": [html.Strong(s) for s in ["Positive (P)", "Negative (N)", "", "", ""]],
    "Predicted Positive (PP)": [
        html.Span(id="tbl-tp"),
        html.Span(id="tbl-fp"),
        html.Span(id="tbl-precision"),
        html.Span(id="tbl-fdr"),
        "",
    ],
    "Predicted Negative (PN)": [
        html.Span(id="tbl-fn"),
        html.Span(id="tbl-tn"),
        html.Span(id="tbl-fomr"),
        html.Span(id="tbl-npv"),
        "",
    ],
    " ": [
        html.Span(id="tbl-recall"),
        html.Span(id="tbl-fpr"),
        "",
        "",
        "",
    ],
    "  ": [
        html.Span(id="tbl-fnr"),
        html.Span(id="tbl-specificity"),
        "",
        "",
        "",
    ],
}
confusion_matrix_table = pd.DataFrame(confusion_matrix_table)

# build the layout
app.layout = dbc.Container(
    [
        html.H1("Classifier Performance Visualizer", className="display-4"),
        html.H2("Define the classifier", className="mb-4"),
        html.P("Initialize a classifier either from performance metrics and a probability distribution"
               "for sampling predictions. Alternative, upload arrays corresponding to `y_true` and `y_pred`."),
        dbc.Col(
            dbc.Stack(
                [
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Precision"),
                            dbc.Input(id="precision-input", value=0.9, type="number", min=0, max=1, step=0.05),
                        ]
                    ),
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Recall"),
                            dbc.Input(id="recall-input", value=0.9, type="number", min=0, max=1, step=0.05),
                        ]
                    ),
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Specificity"),
                            dbc.Input(id="specificity-input", value=0.9, type="number", min=0, max=1, step=0.05),
                        ]
                    ),
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("N"),
                            dbc.Input(id="n-samples-input", value=500, type="number", min=50, max=10000, step=10),
                        ]
                    ),
                ],
            ),
            width=6,
        ),
        html.H2("Adjust Threshold"),  # TODO: add margin class to this
        dbc.Row(
            [
                html.Label("Classifier threshold"),
                dcc.Slider(id="threshold-input", value=0.5, min=0, max=1, tooltip={"always_visible": False}),
            ]
        ),
        dbc.Col(dbc.Table.from_dataframe(confusion_matrix_table, responsive=True), width=12),
        dbc.Col(
            dbc.Row(
                [
                    dbc.Col(
                        [html.H4("Violin plot", className="text-center"), dcc.Graph(id="violins-plot")],
                        width=12,
                        md=6,
                        lg=4,
                    ),
                    dbc.Col(
                        [html.H4("ROC curve", className="text-center"), dcc.Graph(id="roc-curve")], width=12, md=6, lg=4
                    ),
                    dbc.Col(
                        [
                            html.H4("Precision-recall curve", className="text-center"),
                            dcc.Graph(id="precision-recall-curve"),
                        ],
                        width=12,
                        md=6,
                        lg=4,
                    ),
                ]
            ),
            width=12,
        ),
        # storage
        dcc.Store(id="classifier", storage_type="memory"),
    ],
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
    Output("tbl-fomr", "children"),
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
    fig = classifier.plot_violins(engine="plotly")
    fig = fig.update_layout(margin=FIG_MARGIN)
    return fig


@callback(Output("roc-curve", "figure"), Input("classifier", "data"))
def update_roc_curve(classifier_dict: dict):
    classifier = MockClassifier(**classifier_dict)
    fig = classifier.plot_roc_curve()
    fig = fig.update_layout(margin=FIG_MARGIN)
    return fig


@callback(Output("precision-recall-curve", "figure"), Input("classifier", "data"))
def update_precision_recall_curve(classifier_dict: dict):
    classifier = MockClassifier(**classifier_dict)
    fig = classifier.plot_precision_recall_curve()
    fig = fig.update_layout(margin=FIG_MARGIN)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
