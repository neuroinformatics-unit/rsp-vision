import itertools
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import Dash, Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from rsp_vision.dashboard.layout import get_direction_plot_for_controller
from rsp_vision.dashboard.plotting_helpers import (
    find_peak_coordinates,
    fit_correlation,
    get_circle_coordinates,
    get_dataframe_for_facet_plot,
    get_peaks_dataframe,
)
from rsp_vision.objects.photon_data import PhotonData


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


def get_update_fig_all_sessions_callback(
    app: Dash, signal: pd.DataFrame
) -> None:
    @app.callback(
        Output("session-graph", "children"),
        Input("roi-choice-dropdown", "value"),
    )
    def update_fig_all_sessions(roi_id: int) -> dcc.Graph:
        pastel_colors = [
            "#acd0f8",
            "#ace7d0",
        ]

        unique_session_ids = signal["session_id"].unique()
        line_data = []

        for i, session_id in enumerate(unique_session_ids):
            session_data = signal[
                (signal.roi_id == roi_id) & (signal.session_id == session_id)
            ]
            line_data.append(
                go.Scatter(
                    x=session_data["frames_id"],
                    y=session_data["signal"],
                    mode="lines",
                    line=dict(
                        color=pastel_colors[i % len(pastel_colors)], width=0.3
                    ),
                    showlegend=False,
                )
            )

        responses = signal[signal.stimulus_onset]
        scatterplot = px.scatter(
            responses,
            x="frames_id",
            y="sf",
        )

        scatterplot.update_traces(showlegend=False)

        fig = go.Figure(data=line_data + list(scatterplot.data))
        fig.update_layout(
            title=f"Signal across sessions, roi: {roi_id + 1}",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            showlegend=False,
            width=1500,
        )

        return html.Div(
            dcc.Graph(
                id="my-graph",
                figure=fig,
            )
        )


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
    def sf_tf_grid(roi_id: int, dir: dict) -> dcc.Graph:
        dir = dir["value"]
        vertical_df = get_dataframe_for_facet_plot(
            signal, data, counts, roi_id, dir
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
            category_orders={"sf": data._sf[::-1], "tf": data._tf},
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


def get_andermann_gaussian_plot_callback(
    app: Dash,
    median_subtracted_responses: np.ndarray,
    downsampled_gaussians: Dict[Tuple[int, int], np.ndarray],
    oversampled_gaussians: Dict[Tuple[int, int], np.ndarray],
    fit_outputs: np.ndarray,
    sfs: np.ndarray,
    tfs: np.ndarray,
) -> None:
    @app.callback(
        Output("gaussian-graph-andermann", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("direction-store", "data"),
        ],
    )
    def gaussian_plot(roi_id: int, _dir: dict) -> html.Div:
        dir = _dir["value"]

        # Create subplots for the two Gaussian plots
        fig = sp.make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Median subtracted response",
                "Original Gaussian",
                "Oversampled Gaussian",
            ),
        )
        uniform_sfs = uniform_tfs = np.arange(0, len(sfs), 1)

        #  Add the heatmap for the median subtracted response
        fig.add_trace(
            go.Heatmap(
                z=median_subtracted_responses[(roi_id, dir)],
                x=uniform_tfs,
                y=uniform_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=1,
        )

        # Add the heatmap for the original Gaussian
        fig.add_trace(
            go.Heatmap(
                z=downsampled_gaussians[(roi_id, dir)],
                x=uniform_tfs,
                y=uniform_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=2,
        )

        # Add the heatmap for the oversampled Gaussian
        oversampling_factor = 100
        uniform_oversampled_sfs = np.linspace(
            0, oversampling_factor - 1, oversampling_factor
        )
        uniform_oversampled_tfs = np.linspace(
            0, oversampling_factor - 1, oversampling_factor
        )

        fig.add_trace(
            go.Heatmap(
                z=oversampled_gaussians[(roi_id, dir)],
                x=uniform_oversampled_tfs,
                y=uniform_oversampled_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=3,
        )

        log_sfs = np.logspace(
            np.log2(min(sfs)),
            np.log2(max(sfs)),
            num=oversampling_factor,
            base=2,
        )

        log_tfs = np.logspace(
            np.log2(min(tfs)),
            np.log2(max(tfs)),
            num=oversampling_factor,
            base=2,
        )

        fit_corr = fit_correlation(
            downsampled_gaussians[(roi_id, dir)],
            median_subtracted_responses[(roi_id, dir)],
        )

        # Update layout to maintain the aspect ratio
        fig.update_layout(
            autosize=False,
            width=1100,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=False,
            title_text=f"Fit Correlation: {fit_corr:.2f}, \
                𝜁: {fit_outputs[(roi_id, dir)][-1]:.2f}",
        )

        fig.update_xaxes(tickvals=uniform_tfs, ticktext=tfs, row=1, col=1)
        fig.update_yaxes(tickvals=uniform_sfs, ticktext=sfs, row=1, col=1)
        fig.update_xaxes(tickvals=uniform_tfs, ticktext=tfs, row=1, col=2)
        fig.update_yaxes(tickvals=uniform_sfs, ticktext=sfs, row=1, col=2)
        fig.update_yaxes(
            tickvals=uniform_oversampled_sfs[::10],
            ticktext=np.round(log_sfs[::10], 2),
            row=1,
            col=3,
        )
        fig.update_xaxes(
            tickvals=uniform_oversampled_tfs[::10],
            ticktext=np.round(log_tfs[::10], 2),
            row=1,
            col=3,
        )

        fig.update_yaxes(title_text="Spatial Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Temporal Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Spatial Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Temporal Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Spatial Frequency", row=1, col=3)
        fig.update_xaxes(title_text="Temporal Frequency", row=1, col=3)

        return html.Div(
            dcc.Graph(
                id="gaussian_plot",
                figure=fig,
            )
        )


def get_murakami_plot_callback(
    app: Dash,
    n_roi: int,
    directions: List[int],
    sfs: np.ndarray,
    tfs: np.ndarray,
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
        _roi_id: int, _dire: dict, rois_to_show: str, scale: str
    ) -> html.Div:
        _dire = _dire["value"]

        colors = px.colors.qualitative.Light24[:n_roi]

        fig = go.Figure()

        def figure_for_murakami_plot(roi_id):
            color = colors[roi_id]
            peaks = {
                (roi_id, dire): find_peak_coordinates(
                    oversampled_gaussians[(roi_id, dire)], sfs, tfs, config
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

            if roi_id == _roi_id:
                row = p[(p.roi_id == _roi_id) & (p.direction == _dire)].iloc[0]
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
            fig = figure_for_murakami_plot(_roi_id)
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


def get_polar_plot_callback(
    app: Dash,
    directions: List[int],
    sfs: np.ndarray,
    tfs: np.ndarray,
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
            sfs,
            tfs,
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
        colors = px.colors.qualitative.Light24[: int(len(sfs))]

        for i, (tf, sf) in enumerate(itertools.product(tfs, sfs)):
            row = p_sorted[
                (p_sorted.temporal_frequency == tf)
                & (p_sorted.spatial_frequency == sf)
            ]
            color = colors[i % len(sfs)]

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
    sfs: np.ndarray,
    tfs: np.ndarray,
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
        sorted_sfs = sorted(sfs, reverse=True)
        sorted_tfs = sorted(tfs)

        p_sorted = get_peaks_dataframe(
            gaussian_or_original,
            roi_id,
            directions,
            sorted_sfs,
            sorted_tfs,
            median_subtracted_responses,
            downsampled_gaussians,
        )

        max_value = p_sorted["corresponding_value"].max()

        ncols = len(sorted_sfs)
        nrows = len(sorted_tfs)

        subplot_titles = [
            f"sf: {sf}, tf: {tf}"
            for sf, tf in itertools.product(sorted_sfs, sorted_tfs)
        ]
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
            tf = sorted_tfs[tf_idx]
            sf = sorted_sfs[sf_idx]

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
