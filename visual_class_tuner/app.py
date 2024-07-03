from dash import Dash, html, dcc, callback, Output, Input, State, ALL
from dash import callback_context as ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import re

from scipy.stats._continuous_distns import _distn_names as cont_distns
from scipy.stats._discrete_distns import _distn_names as disc_distns
from scipy import stats

# random number generator
rng = np.random.default_rng()

# create the app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# build the layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dbc.InputGroupText("Select distribution"), width={"size": 2}),
                dbc.Col(
                    html.Div(dcc.Dropdown(options=cont_distns + disc_distns, value="norm", id="dropdown-selection"))
                ),
            ],
            className="g-0",
        ),
        html.H1("General parameters"),
        dbc.Stack(
            children=[
                dbc.InputGroup([dbc.InputGroupText("size"), dbc.Input(id="distn-size", value=1000, type="number")]),
                dbc.InputGroup([dbc.InputGroupText("loc"), dbc.Input(id="distn-loc", value=0, type="number")]),
                dbc.InputGroup([dbc.InputGroupText("scale"), dbc.Input(id="distn-scale", value=1, type="number")]),
                dbc.InputGroup([dbc.InputGroupText("nbins"), dbc.Input(id="distn-nbins", value=None, type="number")]),
            ],
            id="general-distn-params",
        ),
        html.Hr(),
        html.H1("Specific parameters"),
        dbc.Stack(children=None, id="spec-distn-params"),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id="graph-content")),
            dbc.Col(dbc.Card(id="distn-doc", body=True)),
        ]),
        dbc.Alert(children=None, id="alert-message", color="primary"),
        # storage
        dcc.Store(id="distn-params", storage_type="memory"),
    ],
    fluid=True,
)


@callback(Output("spec-distn-params", "children"), Input("dropdown-selection", "value"))
def add_distn_params_to_layout(distn_name):
    """Dynamically create input elements for each parameter depending on the distribution."""
    func_gen = getattr(stats, distn_name)
    if func_gen.shapes is not None:
        shapes = func_gen.shapes.split(", ")
        children = [
            dbc.InputGroup(
                [
                    dbc.InputGroupText(shape),
                    dbc.Input(id={"type": "distn-param", "index": shape}, value=PARAM_DEFAULTS[shape], type="number"),
                ]
            )
            for shape in shapes
        ]
        return children


def _validate_distn_params(distn_params: dict) -> dict:
    # get the generator for the distribution sampler
    func_gen = getattr(stats, distn_params["name"])

    # check that the sampler can be created from the generator
    try:
        func = func_gen(**{k: v for k, v in distn_params.items() if k not in ["name", "nbins", "size"]})
    except TypeError as e:
        if "_parse_args() missing 1 required positional argument" in str(e):
            raise
        if BAD_ARG_MESSAGE in str(e):
            for arg in ["scale", "loc"]:
                if f"{BAD_ARG_MESSAGE} '{arg}'" == str(e):
                    del distn_params[arg]
            try:
                func = func_gen(**{k: v for k, v in distn_params.items() if k not in ["name", "nbins", "size"]})
            except TypeError:
                raise

    # check that random variable samples can be generater
    try:
        _ = func.rvs(size=distn_params["size"], random_state=rng)
    except (TypeError, ValueError):
        raise
    return distn_params


@callback(
    Output("distn-params", "data"),
    Output("alert-message", "children"),
    State("dropdown-selection", "value"),
    Input("distn-size", "value"),
    Input("distn-loc", "value"),
    Input("distn-scale", "value"),
    Input("distn-nbins", "value"),
    Input({"type": "distn-param", "index": ALL}, "value"),
    Input({"type": "distn-param", "index": ALL}, "id"),
)
def update_distn_params(
    distn_name, distn_size, distn_loc, distn_scale, distn_nbins, distn_param_values, distn_param_ids
):
    distn_params = {
        "name": distn_name,
        "size": distn_size,
        "loc": distn_loc,
        "scale": distn_scale,
        "nbins": distn_nbins,
    }
    spec_distn_params = {}
    for param_id, param_value in zip(distn_param_ids, distn_param_values):
        if param_value is None:
            return go.Figure(), "Not all params selected"
        spec_distn_params.setdefault(param_id["index"], param_value)
    distn_params.update(spec_distn_params)
    try:
        distn_params = _validate_distn_params(distn_params)
    except (ValueError, TypeError) as e:
        return {}, str(e)
    return distn_params, None


@callback(
    Output("graph-content", "figure"),
    Output("distn-doc", "children"),
    Input("distn-params", "data")
)
def update_graph(distn_params):
    if distn_params == {}:
        return go.Figure(), ""
    distn_name = distn_params.pop("name")
    nbins = distn_params.pop("nbins")
    size = distn_params.pop("size")
    func_gen = getattr(stats, distn_name)
    func = func_gen(**distn_params)
    data_sample = func.rvs(size=size, random_state=rng)
    doc_string = re.split(r"\n\s*\n", func_gen.__doc__)
    doc_string = [html.P(p.strip()) for p in doc_string]
    return px.histogram(data_sample, nbins=nbins), doc_string


if __name__ == "__main__":
    app.run(debug=True)
