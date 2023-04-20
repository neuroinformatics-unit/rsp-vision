import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import Dash, Input, Output, dcc, html

from rsp_vision.dashboard.query_dataframes import (
    find_peak_coordinates,
    fit_correlation,
    get_dataframe_for_facet_plot,
)


def get_update_fig_all_sessions_callback(app: Dash, signal) -> None:
    @app.callback(
        Output("example-graph", "children"), Input("demo-dropdown", "value")
    )
    def update_fig_all_sessions(roi_id):
        pastel_colors = [
            "#acd0f8", "#ace7d0", 
        ]

        unique_session_ids = signal["session_id"].unique()
        line_data = []

        for i, session_id in enumerate(unique_session_ids):
            session_data = signal[(signal.roi_id == roi_id) & (signal.session_id == session_id)]
            line_data.append(
                go.Scatter(
                    x=session_data["frames_id"],
                    y=session_data["signal"],
                    mode="lines",
                    line=dict(color=pastel_colors[i % len(pastel_colors)], width=0.3),
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
            Input("demo-dropdown", "value"),
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
            Input("demo-dropdown", "value"),
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
            Input("demo-dropdown", "value"),
            Input("directions-checkbox", "value"),
            Input("which-roi-to-show-in-murakami-plot", "value"),
            Input("murakami-plot-scale", "value"),
        ],
    )
    def murakami_plot(_roi_id, _dire, rois_to_show, scale):
        fig = go.Figure()

        #  range of plotly colors equal to the length of n_roi
        colors = colors = px.colors.qualitative.Light24[:n_roi]

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
            # yaxis_type="log",
            # xaxis_type="log",
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
