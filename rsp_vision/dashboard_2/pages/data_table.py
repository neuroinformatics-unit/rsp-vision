from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, callback, dash_table
from decouple import config

from rsp_vision.load.load_data import read_config_file
from rsp_vision.objects.SWC_Blueprint import (
    SessionFolder,
    SubjectFolder,
    SWC_Blueprint_Spec,
)

CONFIG_PATH = config("CONFIG_PATH")
config_path = Path(__file__).parents[2] / CONFIG_PATH
config = read_config_file(config_path)
swc_blueprint_spec = SWC_Blueprint_Spec(
    project_name="rsp_vision",
    raw_data=False,
    derivatives=True,
    local_path=Path(config["paths"]["output"]),
)
with open(swc_blueprint_spec.path / "analysis_log.csv", "r") as f:
    c = f.readline().split(",")
    c[0] = "index"
    columns = [{"name": i, "id": i} for i in c]
    dataframe = pd.read_csv(f, names=c, index_col=0)
    data = dataframe.to_dict(orient="records")

dash.register_page(__name__, path="/")


layout = dash.html.Div(
    [
        dmc.Title(
            "Data Table",
            order=2,
            className="page-title",
        ),
        dmc.Grid(
            children=[
                dmc.Text(
                    id="selected_data_str",
                    className="selected-data-text",
                ),
                dbc.Button(
                    "Load Data",
                    id="button",
                    className="load-data-button",
                    href="/murakami_plot",
                    n_clicks=0,
                ),
            ],
            className="selected-data-container",
        ),
        dmc.Container(
            children=[
                dash_table.DataTable(
                    id="table",
                    columns=columns,
                    data=data,
                    editable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    row_selectable="single",
                    row_deletable=False,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=20,
                    hidden_columns=[
                        "index",
                        "sub",
                        "ses",
                        "mouse line",
                        "mouse id",
                        "hemisphere",
                        "brain region",
                        "monitor position",
                        "fov",
                        "cre",
                        "analysed",
                        "commit hash",
                        "microscope",
                    ],
                ),
            ],
            className="table",
        ),
    ],
    className="page",
)


@callback(
    [
        Output("selected_data_str", "children"),
        Output("store", "data"),
        Output("button", "disabled"),
    ],
    Input("table", "selected_rows"),
)
def update_graphs(selected_rows):
    if selected_rows is None or len(selected_rows) == 0:
        return "Select data to be loaded", {}, True
    else:
        sub_folder = SubjectFolder(
            swc_blueprint_spec,
            dataframe.iloc[selected_rows[0]].to_dict(),
            sub_num=0,  # irrelevant
        )
        print(sub_folder)
        session_folder = SessionFolder(
            sub_folder,
            dataframe.iloc[selected_rows[0]].to_dict(),
            ses_num=0,  # irrelevant
        )
        print(session_folder)

        store = {
            "data": dataframe.iloc[selected_rows[0]],
            "path": str(swc_blueprint_spec.path),
            "config": config,
            "oversampling_factor": int(
                config["fitting"]["oversampling_factor"]
            ),
            "subject_folder_path": str(sub_folder.sub_folder_path),
            "session_folder_path": str(session_folder.ses_folder_path),
        }
        folder_name = dataframe.iloc[selected_rows[0]]["folder name"]
        return (
            f"Selected data: {folder_name}",
            store,
            False,
        )
