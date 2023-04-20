import itertools
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from dash import Dash, Input, Output, dcc, html

from rsp_vision.dashboard.callbacks.plotting_helpers import (
    get_dataframe_for_facet_plot,
)
from rsp_vision.objects.photon_data import PhotonData


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

        return html.Div(
            dcc.Graph(
                id="sf_tf_plot",
                figure=fig,
            )
        )
