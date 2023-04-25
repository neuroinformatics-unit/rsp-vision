import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from rsp_vision.dashboard.callbacks.plotting_helpers import (
    get_dataframe_for_facet_plot,
    get_dataframe_for_facet_plot_pooled_directions,
)
from rsp_vision.objects.photon_data import PhotonData


def single_direction_plot(
    signal: np.ndarray,
    data: PhotonData,
    roi_id: int,
    direction: str,
    counts: pd.DataFrame,
) -> dcc.Graph:
    assert isinstance(direction, int)
    vertical_df = get_dataframe_for_facet_plot(
        signal, data, counts, roi_id, direction
    )

    fig = px.line(
        vertical_df,
        x="stimulus_frames",
        y="signal",
        facet_col="tf",
        facet_row="sf",
        width=1500,
        height=800,
        color="signal_kind",
        category_orders={
            "sf": data.spatial_frequencies[::-1],
            "tf": data.temporal_frequencies,
        },
    )

    fig.update_layout(
        title=f"SF TF traces for roi {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    fig.update_traces(line=dict(width=0.5))

    return fig


def all_directions_plot(
    signal: np.ndarray,
    data: PhotonData,
    roi_id: int,
) -> dcc.Graph:
    combined_df = get_dataframe_for_facet_plot_pooled_directions(
        signal, roi_id
    )

    fig = px.line(
        combined_df,
        x="stimulus_frames",
        y="signal",
        facet_col="tf",
        facet_row="sf",
        width=1500,
        height=800,
        color="session_direction",
        category_orders={
            "sf": data.spatial_frequencies[::-1],
            "tf": data.temporal_frequencies,
        },
    )
    for trace in fig.data:
        if trace.name == "mean":
            trace.line.color = "black"
            trace.line.width = 3
            trace.line.dash = "solid"
        elif trace.name == "median":
            trace.line.color = "red"
            trace.line.width = 3
            trace.line.dash = "solid"
        else:
            trace.line.width = 0.5

    fig.update_layout(
        title=f"SF TF traces for roi {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
    )

    fig.add_annotation(
        x=0.9,
        y=0.97,
        xref="paper",
        yref="paper",
        text="mean",
        showarrow=False,
        font=dict(size=15, color="black"),
    )

    fig.add_annotation(
        x=0.95,
        y=0.97,
        xref="paper",
        yref="paper",
        text="median",
        showarrow=False,
        font=dict(size=15, color="red"),
    )

    return fig


def get_sf_tf_grid_callback(
    app: Dash, signal: pd.DataFrame, data: PhotonData, counts: np.ndarray
) -> None:
    @app.callback(
        Output("sf_tf-graph", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("direction-store", "data"),
        ],
    )
    def sf_tf_grid(roi_id: int, direction_input: dict) -> dcc.Graph:
        direction = direction_input["value"]
        if direction == "all":
            fig = all_directions_plot(signal, data, roi_id)
        else:
            fig = single_direction_plot(
                signal, data, roi_id, direction, counts
            )

        return html.Div(
            dcc.Graph(
                id="sf_tf_plot",
                figure=fig,
            )
        )
