import os
from os.path import isfile, join

import dash
from dash import Input, Output, callback, dcc, html

dash.register_page(
    __name__,
    path="/",
    # redirect_from=["/home/"],
)

path_to_data = "/Users/laura/data/output"


def get_list_of_datasets(path_to_data):
    onlyfiles = [
        f for f in os.listdir(path_to_data) if isfile(join(path_to_data, f))
    ]
    # remove extension
    onlyfiles = [f.split(".")[0] for f in onlyfiles]
    return onlyfiles


datasets = get_list_of_datasets(path_to_data)


@callback(
    [Output("redirect", "pathname"), Output("redirect", "search")],
    [Input("load-button", "n_clicks")],
    [dash.dependencies.State("dataset-dropdown", "value")],
)
def update_dashboard(n_clicks, selected_dataset):
    if n_clicks > 0:
        return "/dashboard", f"?data={selected_dataset}"

    return None, None


layout = html.Div(
    [
        html.H1("Select Dataset to Load"),
        dcc.Dropdown(
            id="dataset-dropdown",
            options=[{"label": ds, "value": ds} for ds in datasets],
            placeholder="Select a dataset",
        ),
        html.Button("Load", id="load-button", n_clicks=0),
        dcc.Location(id="redirect", pathname=None, search=None, refresh=True),
    ]
)
