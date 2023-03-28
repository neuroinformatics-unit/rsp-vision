import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from rsp_vision.dashboard.query_dataframes import (
    find_peak_coordinates,
    fit_correlation,
    get_dataframe_for_facet_plot,
)


def get_update_fig_all_sessions_callback(app: Dash, signal) -> None:
    @app.callback(
        Output("example-graph", "children"),
        Input("roi-choice-dropdown", "value"),
    )
    def update_fig_all_sessions(roi_id):
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

        scatterplot = px.scatter(
            signal[signal.stimulus_onset],
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
        )

        return html.Div(
            dcc.Graph(
                id="my-graph",
                figure=fig,
            )
        )


def get_sf_tf_grid_callback(app: Dash, signal, data, counts) -> None:
    @app.callback(
        Output("sf_tf-graph", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("directions-checkbox", "value"),
        ],
    )
    def sf_tf_grid(roi_id, dir):
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
    median_subtracted_responses,
    downsapled_gaussians,
    oversampled_gaussians,
    fit_outputs,
    sfs,
    tfs,
) -> None:
    @app.callback(
        Output("gaussian-graph-andermann", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("directions-checkbox", "value"),
        ],
    )
    def gaussian_plot(roi_id, dir):
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
        uniform_sfs = np.linspace(0, len(sfs) - 1, len(sfs))
        uniform_tfs = np.linspace(0, len(tfs) - 1, len(tfs))

        #  Add the heatmap for the median subtracted response
        fig.add_trace(
            go.Heatmap(
                z=median_subtracted_responses[(roi_id, dir)],
                x=uniform_sfs,
                y=uniform_tfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=1,
        )

        fig.update_xaxes(tickvals=uniform_sfs, ticktext=sfs, row=1, col=1)
        fig.update_yaxes(tickvals=uniform_tfs, ticktext=tfs, row=1, col=1)

        # Add the heatmap for the original Gaussian
        fig.add_trace(
            go.Heatmap(
                z=downsapled_gaussians[(roi_id, dir)],
                x=uniform_sfs,
                y=uniform_tfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=2,
        )
        # Update the tick labels to display the original values
        fig.update_xaxes(tickvals=uniform_sfs, ticktext=sfs, row=1, col=2)
        fig.update_yaxes(tickvals=uniform_tfs, ticktext=tfs, row=1, col=2)

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
                x=uniform_oversampled_sfs,
                y=uniform_oversampled_tfs,
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

        fig.update_xaxes(
            tickvals=uniform_oversampled_sfs[::10],
            ticktext=np.round(log_sfs[::10], 2),
            row=1,
            col=3,
        )
        fig.update_yaxes(
            tickvals=uniform_oversampled_tfs[::10],
            ticktext=np.round(log_tfs[::10], 2),
            row=1,
            col=3,
        )

        fit_corr = fit_correlation(
            downsapled_gaussians[(roi_id, dir)],
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

        # Update axis titles
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


def get_labels(data, sf_inverted=True):
    if sf_inverted:
        y_labels = list(map(str, data._sf[::-1].tolist()))
    else:
        y_labels = list(map(str, data._sf.tolist()))
    x_labels = list(map(str, data._tf.tolist()))

    return x_labels, y_labels


def get_murakami_plot_callback(
    app: Dash,
    n_roi,
    directions,
    sfs,
    tfs,
    oversampled_gaussians,
    responsive_rois,
) -> None:
    @app.callback(
        Output("murakami-plot", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("directions-checkbox", "value"),
            Input("which-roi-to-show-in-murakami-plot", "value"),
            Input("murakami-plot-scale", "value"),
        ],
    )
    def murakami_plot(_roi_id, _dire, rois_to_show, scale):
        fig = go.Figure()

        #  range of plotly colors equal to the length of n_roi
        colors = px.colors.qualitative.Light24[:n_roi]

        def murakami_plot_tools(roi_id):
            color = colors[roi_id]
            peaks = {
                (roi_id, dire): find_peak_coordinates(
                    oversampled_gaussians[(roi_id, dire)], sfs, tfs
                )
                # for roi_id in range(n_roi)
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

            # # Add median point for each ROI
            median_peaks = p.groupby("roi_id").median(
                ["temporal_frequency", "spatial_frequency"]
            )

            # connect with a line all blue dots with the
            # red dot (median) one by one
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

        if rois_to_show == "choosen":
            murakami_plot_tools(_roi_id)
        elif rois_to_show == "responsive":
            for roi_id in responsive_rois:
                murakami_plot_tools(roi_id)
        elif rois_to_show == "all":
            for roi_id in range(n_roi):
                murakami_plot_tools(roi_id)

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


def get_corresponding_value(data, roi_id, dire, sf_idx, tf_idx):
    # if I use the oversampled gaussian, I get a different result
    # there is always a point in which the peak is very high
    # therefore it does not give us much information on the preference
    # of the neuron
    matrix = data[(roi_id, dire)]
    return matrix[tf_idx, sf_idx]


def get_peaks_dataframe(
    gaussian_or_original,
    roi_id,
    directions,
    sfs,
    tfs,
    median_subtracted_responses,
    downsampled_gaussians,
):
    if gaussian_or_original == "original":
        data = median_subtracted_responses
    elif gaussian_or_original == "gaussian":
        data = downsampled_gaussians

    p = pd.DataFrame(
        {
            "roi_id": roi_id,
            "direction": dire,
            "temporal_frequency": tfs[tf_idx],
            "spatial_frequency": sfs[sf_idx],
            "corresponding_value": get_corresponding_value(
                data, roi_id, dire, sf_idx, tf_idx
            ),
        }
        for dire in directions
        for tf_idx, sf_idx in itertools.product(
            range(len(tfs)), range(len(sfs))
        )
    )

    p_sorted = p.sort_values(by="direction")

    return p_sorted


def get_polar_plot_callback(
    app: Dash,
    directions,
    sfs,
    tfs,
    downsampled_gaussians,
    median_subtracted_responses,
) -> None:
    @app.callback(
        Output("polar-plot", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("polar-plot-gaussian-or-original", "value"),
            Input("polar-plot-mean-or-median-or-cumulative", "value"),
        ],
    )
    def polar_plot(roi_id, gaussian_or_original, mean_or_median_or_cumulative):
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
    directions,
    sfs,
    tfs,
    downsampled_gaussians,
    median_subtracted_responses,
) -> None:
    @app.callback(
        Output("polar-plot-facet", "children"),
        [
            Input("roi-choice-dropdown", "value"),
            Input("polar-plot-gaussian-or-original", "value"),
        ],
    )
    def polar_plot_facet(roi_id, gaussian_or_original):
        p_sorted = get_peaks_dataframe(
            gaussian_or_original,
            roi_id,
            directions,
            sfs,
            tfs,
            median_subtracted_responses,
            downsampled_gaussians,
        )

        max_value = p_sorted["corresponding_value"].max()

        ncols = len(sfs)
        nrows = len(tfs)

        subplot_titles = [
            f"sf: {sf}, tf: {tf}" for tf, sf in itertools.product(tfs, sfs)
        ]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=[[{"type": "polar"}] * ncols] * nrows,
            horizontal_spacing=0.04,
            vertical_spacing=0.04,
            subplot_titles=subplot_titles,
        )

        for tf_idx, sf_idx in itertools.product(
            range(len(tfs)), range(len(sfs))
        ):
            tf = tfs[tf_idx]
            sf = sfs[sf_idx]

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
                row=tf_idx + 1,
                col=sf_idx + 1,
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
