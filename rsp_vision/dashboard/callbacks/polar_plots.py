import itertools
from typing import Dict, List, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from rsp_vision.dashboard.callbacks.plotting_helpers import get_peaks_dataframe


def get_polar_plot_callback(
    app: Dash,
    directions: List[int],
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    downsampled_gaussians: Dict[Tuple[int, int], np.ndarray],
    median_subtracted_responses: Dict[Tuple[int, int], np.ndarray],
) -> None:
    @app.callback(
        Output("polar-plot", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("polar-plot-gaussian-or-original", "value"),
            Input("polar-plot-mean-or-median-or-cumulative", "value"),
        ],
    )
    def polar_plot(
        roi_id: int,
        gaussian_or_original: str,
        mean_or_median_or_cumulative: str,
    ) -> html.Div:
        p_sorted = get_peaks_dataframe(
            gaussian_or_original,
            roi_id,
            directions,
            spatial_frequencies,
            temporal_frequencies,
            median_subtracted_responses,
            downsampled_gaussians,
        )

        pivot_table = p_sorted.pivot(
            index=["temporal_frequency", "spatial_frequency"],
            columns="direction",
            values="corresponding_value",
        )

        subplot_titles = [
            "Responses across sf/tf for each direction",
            "",
        ]
        if mean_or_median_or_cumulative == "mean":
            total_values = pivot_table.mean(axis=0)
            subplot_titles[1] = "mean responses"
        elif mean_or_median_or_cumulative == "median":
            total_values = pivot_table.median(axis=0)
            subplot_titles[1] = "median responses"
        elif mean_or_median_or_cumulative == "cumulative":
            pivot_table = pivot_table - pivot_table.min().min()
            pivot_table = pivot_table / pivot_table.max().max()
            total_values = pivot_table.sum(axis=0)
            subplot_titles[1] = "cumulative responses"

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "polar"}, {"type": "polar"}]],
            subplot_titles=subplot_titles,
        )
        colors = px.colors.qualitative.Light24[: int(len(spatial_frequencies))]

        for i, (tf, sf) in enumerate(
            itertools.product(temporal_frequencies, spatial_frequencies)
        ):
            row = p_sorted[
                (p_sorted.temporal_frequency == tf)
                & (p_sorted.spatial_frequency == sf)
            ]
            color = colors[i % len(spatial_frequencies)]

            r_values = row["corresponding_value"].tolist()
            theta_values = row["direction"].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=r_values + [r_values[0]],
                    theta=theta_values + [theta_values[0]],
                    mode="lines",
                    thetaunit="degrees",
                    line=dict(color=color, width=4),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatterpolar(
                r=total_values.tolist() + [total_values.tolist()[0]],
                theta=total_values.index.tolist()
                + [total_values.index.tolist()[0]],
                mode="lines",
                thetaunit="degrees",
                name="Cumulative",
                line=dict(color="black", width=3),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            width=1600,
            height=800,
            title=dict(
                text="Analysis of responses across sf/tf for each direction",
                font=dict(size=24, color="black", family="Arial, bold"),
                xanchor="center",
                yanchor="top",
                x=0.5,
                y=1,
            ),
            annotations=[
                dict(
                    font=dict(size=18, color="black", family="Arial, bold"),
                    xanchor="center",
                    yanchor="top",
                    x=0.22,
                    y=1.08,
                    showarrow=False,
                ),
                dict(
                    font=dict(size=18, color="black", family="Arial, bold"),
                    xanchor="center",
                    yanchor="top",
                    x=0.78,
                    y=1.08,
                    showarrow=False,
                ),
            ],
        )

        return html.Div(
            dcc.Graph(
                id="polar_plot",
                figure=fig,
            )
        )


def get_polar_plot_facet_callback(
    app: Dash,
    directions: List[int],
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    downsampled_gaussians: Dict[Tuple[int, int], np.ndarray],
    median_subtracted_responses: Dict[Tuple[int, int], np.ndarray],
) -> None:
    @app.callback(
        Output("polar-plot-facet", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("polar-plot-gaussian-or-original", "value"),
        ],
    )
    def polar_plot_facet(roi_id: int, gaussian_or_original: str) -> html.Div:
        sorted_sfs = sorted(spatial_frequencies, reverse=True)
        sorted_tfs = sorted(temporal_frequencies)

        p_sorted = get_peaks_dataframe(
            gaussian_or_original=gaussian_or_original,
            roi_id=roi_id,
            directions=directions,
            spatial_frequencies=sorted_sfs,
            temporal_frequencies=sorted_tfs,
            median_subtracted_responses=median_subtracted_responses,
            downsampled_gaussians=downsampled_gaussians,
        )

        max_value = p_sorted["corresponding_value"].max()

        ncols = len(sorted_sfs)
        nrows = len(sorted_tfs)

        subplot_titles = [
            f"sf: {sf}, tf: {tf}"
            for sf, tf in itertools.product(sorted_sfs, sorted_tfs)
        ]
        print(subplot_titles)
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=[[{"type": "polar"}] * ncols] * nrows,
            horizontal_spacing=0.04,
            vertical_spacing=0.04,
            subplot_titles=subplot_titles,
        )

        for sf_idx, tf_idx in itertools.product(
            range(len(sorted_sfs)), range(len(sorted_tfs))
        ):
            print("__________")
            print(sf_idx, tf_idx)
            tf = sorted_tfs[tf_idx]
            sf = sorted_sfs[sf_idx]
            print(sf, tf)

            subset = p_sorted[
                (p_sorted["temporal_frequency"] == tf)
                & (p_sorted["spatial_frequency"] == sf)
            ]

            r_values = subset["corresponding_value"].tolist()
            theta_values = subset["direction"].tolist()

            fig.add_trace(
                go.Scatterpolar(
                    r=r_values + [r_values[0]],
                    theta=theta_values + [theta_values[0]],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(0, 0, 0, 0.2)",
                    marker=dict(size=10),
                    line=dict(width=1),
                    thetaunit="degrees",
                    showlegend=False,
                    subplot=f"polar{tf_idx * ncols + sf_idx + 1}",
                ),
                row=sf_idx + 1,
                col=tf_idx + 1,
            )
            print(f"row: {sf_idx + 1}, col: {tf_idx + 1}")

            subplot_name = f"polar{tf_idx * ncols + sf_idx + 1}"
            fig.update_layout(
                {
                    subplot_name: dict(
                        radialaxis=dict(
                            visible=False,
                            range=[0, max_value],
                            gridcolor="rgba(0, 0, 0, 0)",
                        ),
                        angularaxis=dict(
                            visible=False,
                            showticklabels=False,
                            showgrid=False,
                            gridcolor="rgba(0, 0, 0, 0)",
                        ),
                    )
                }
            )

        fig.update_layout(
            title=f"SF TF polar plot for roi {roi_id + 1}",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            width=1500,
            height=1500,
        )

        return html.Div(
            dcc.Graph(
                id="polar-plot-facet-internal",
                figure=fig,
            )
        )
