import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback

dash.register_page(__name__, path="/murakami_plot")

layout = dash.html.Div(
    [
        dmc.Title(
            "Murakami",
            order=2,
            className="page-title",
        ),
        dmc.Text(
            id="stored_data",
            className="selected-data-text",
        ),
    ],
    className="page",
)

@callback(
    Output("stored_data", "children"),
    Input("store", "data"),
)
def read_stored_data(data):
    if data is None:
        return "No data stored"
    return str(data) + "***"
