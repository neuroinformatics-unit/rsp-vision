import os
from os.path import isfile, join

import dash
from dash import Input, Output, callback, dcc, html

dash.register_page(__name__)

path_to_data = "rsp_vision/data"


def get_list_of_datasets(path_to_data):
    onlyfiles = [
        f for f in os.listdir(path_to_data) if isfile(join(path_to_data, f))
    ]
    return onlyfiles


datasets = get_list_of_datasets(path_to_data)

layout = html.Div(
    [
        html.H1("Select Dataset to Load"),
        dcc.Dropdown(
            id="dataset-dropdown",
            options=[{"label": ds, "value": ds} for ds in datasets],
            placeholder="Select a dataset",
        ),
        html.Button("Load", id="load-button", n_clicks=0),
        dcc.Location(id="redirect", refresh=True),
    ]
)


@callback(
    Output("redirect", "pathname"),
    [Input("load-button", "n_clicks")],
    [dash.dependencies.State("dataset-dropdown", "value")],
)
def update_dashboard(n_clicks, selected_dataset):
    if n_clicks > 0:
        # redirect to dashboard and pass the name of the dataset
        return f"/dashboard/?data={selected_dataset}"

    return None
