from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from rsp_vision.dashboard.callbacks.plotting_helpers import (
    find_peak_coordinates,
)


def get_murakami_plot_callback(
    app: Dash,
    n_roi: int,
    directions: List[int],
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    oversampled_gaussians: Dict[Tuple[int, int], np.ndarray],
    responsive_rois: Set[int],
    config: dict,
) -> None:
    @app.callback(
        Output("murakami-plot", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("direction-store", "data"),
            Input("which-roi-to-show-in-murakami-plot", "value"),
            Input("murakami-plot-scale", "value"),
        ],
    )
    def murakami_plot(
        roi_id_input: int, direction_input: dict, rois_to_show: str, scale: str
    ) -> html.Div:
        direction_input = direction_input["value"]

        colors = px.colors.qualitative.Light24[:n_roi]

        fig = go.Figure()

        def figure_for_murakami_plot(roi_id):
            color = colors[roi_id]
            peaks = {
                (roi_id, dire): find_peak_coordinates(
                    oversampled_gaussians[(roi_id, dire)],
                    spatial_frequencies,
                    temporal_frequencies,
                    config,
                )
                for dire in directions
            }

            p = pd.DataFrame(
                {
                    "roi_id": roi_id,
                    "direction": dire,
                    "temporal_frequency": peaks[(roi_id, dire)][0],
                    "spatial_frequency": peaks[(roi_id, dire)][1],
                }
                for dire in directions
            )

            median_peaks = p.groupby("roi_id").median(
                ["temporal_frequency", "spatial_frequency"]
            )

            for i, d in enumerate(directions):
                row = p[(p.roi_id == roi_id) & (p.direction == d)].iloc[0]
                tf = row["temporal_frequency"]
                sf = row["spatial_frequency"]
                fig.add_trace(
                    go.Scatter(
                        x=[tf, median_peaks["temporal_frequency"][roi_id]],
                        y=[sf, median_peaks["spatial_frequency"][roi_id]],
                        mode="markers",
                        marker=dict(color=color, size=10),
                        name=f"ROI {roi_id + 1}" if i == 0 else "",
                        showlegend=True if i == 0 else False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[tf, median_peaks["temporal_frequency"][roi_id]],
                        y=[sf, median_peaks["spatial_frequency"][roi_id]],
                        mode="lines",
                        line=dict(color=color, width=1),
                        showlegend=False,
                    )
                )

            if roi_id == roi_id_input:
                row = p[
                    (p.roi_id == roi_id_input)
                    & (p.direction == direction_input)
                ].iloc[0]
                tf = row["temporal_frequency"]
                sf = row["spatial_frequency"]

                fig.add_trace(
                    go.Scatter(
                        x=[tf],
                        y=[sf],
                        mode="markers",
                        marker=dict(color="red", size=20),
                        showlegend=False,
                    )
                )

            return fig

        if rois_to_show == "choosen":
            fig = figure_for_murakami_plot(roi_id_input)
        elif rois_to_show == "responsive":
            for roi_id in responsive_rois:
                fig = figure_for_murakami_plot(roi_id)
        elif rois_to_show == "all":
            for roi_id in range(n_roi):
                fig = figure_for_murakami_plot(roi_id)

        fig.update_layout(
            title="Murakami plot",
            yaxis_title="Spatial frequency (cycles/deg)",
            xaxis_title="Temporal frequency (Hz)",
            legend_title="ROI",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            autosize=False,
            width=1200,
            height=1000,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        if scale == "log":
            fig.update_yaxes(type="log")
            fig.update_xaxes(type="log")
        else:
            fig.update_yaxes(type="linear")
            fig.update_xaxes(type="linear")

        return html.Div(
            dcc.Graph(
                id="gaussian_plot",
                figure=fig,
            )
        )
