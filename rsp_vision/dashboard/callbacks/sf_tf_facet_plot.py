import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html

from rsp_vision.dashboard.callbacks.plotting_helpers import (
    get_dataframe_for_facet_plot,
    get_dataframe_for_facet_plot_pooled_directions,
)
from rsp_vision.objects.photon_data import PhotonData


def get_sf_tf_grid_callback(
    signal: pd.DataFrame, data: PhotonData, counts: np.ndarray
) -> None:
    @callback(
        Output("sf_tf-graph", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("direction-store", "data"),
            Input("toggle-traces", "value"),
        ],
    )
    def sf_tf_grid(
        roi_id: int, direction_input: dict, toggle_value: str
    ) -> dcc.Graph:
        direction = direction_input["value"]
        if direction == "all":
            dataframe = get_dataframe_for_facet_plot_pooled_directions(
                signal, roi_id
            )
        else:
            assert isinstance(direction, int)
            dataframe = get_dataframe_for_facet_plot(
                signal, data, counts, roi_id, direction
            )

        fig = px.line(
            dataframe,
            x="stimulus_frames",
            y="signal",
            facet_col="tf",
            facet_row="sf",
            facet_col_spacing=0.005,
            facet_row_spacing=0.005,
            width=2000,
            height=1200,
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
            showlegend=False,
        )
        for trace in fig.data:
            if "mean" in trace.name:
                trace.line.color = "black"
                trace.line.width = 3
                trace.line.dash = "solid"
            elif "median" in trace.name:
                trace.line.color = "red"
                trace.line.width = 3
                trace.line.dash = "solid"
            else:
                if "ALL" not in toggle_value:
                    trace.visible = False
                else:
                    trace.line.width = 0.5

        for x0, x1, text, color in [
            (0, 75, "gray", "green"),
            (75, 150, "static", "pink"),
            (150, 225, "drift", "blue"),
        ]:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                row="all",
                col="all",
                annotation_text=text,
                annotation_position="top left",
                annotation_font_size=15,
                fillcolor=color,
                opacity=0.1,
                line_width=0,
            )

        # Fake legend
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

        return html.Div(
            dcc.Graph(
                id="sf_tf_plot",
                figure=fig,
            )
        )
