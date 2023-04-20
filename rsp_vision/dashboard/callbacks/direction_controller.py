import plotly.graph_objects as go
from dash import Dash, Input, Output, State

from rsp_vision.dashboard.callbacks.plotting_helpers import (
    get_circle_coordinates,
)
from rsp_vision.dashboard.layout import get_direction_plot_for_controller


def get_update_circle_figure_callback(app: Dash, directions: list) -> None:
    @app.callback(
        Output("directions-circle", "figure"),
        Input("selected-direction", "children"),
    )
    def update_circle_figure(selected_direction: int) -> go.Figure:
        circle_x, circle_y = get_circle_coordinates(directions)
        return get_direction_plot_for_controller(
            directions, circle_x, circle_y, selected_direction
        )


def get_update_radio_items_callback(app: Dash) -> None:
    @app.callback(
        [
            Output("direction-store", "data"),
            Output("selected-direction", "children"),
        ],
        Input("directions-circle", "clickData"),
        State("direction-store", "data"),
    )
    def update_radio_items(clickData: dict, current_data: dict) -> tuple:
        if clickData is not None:
            direction = clickData["points"][0]["customdata"]
            return {"value": direction}, direction
        else:
            return current_data, current_data["value"]
