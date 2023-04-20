import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import Dash, Input, Output, dcc, html

from rsp_vision.dashboard.query_dataframes import (
    find_peak_coordinates,
    fit_correlation,
    fit_symmetric_gaussian,
    get_dataframe_for_facet_plot,
    get_median_subtracted_response,
)


def get_update_fig_all_sessions_callback(app: Dash, signal) -> None:
    @app.callback(
        Output("example-graph", "children"), Input("demo-dropdown", "value")
    )
    def update_fig_all_sessions(roi_id):
        lineplot = px.line(
            signal[signal.roi_id == roi_id],
            x="frames_id",
            y="signal",
            color="session_id",
        )
        lineplot.update_traces(line=dict(width=0.3))

        scatterplot = px.scatter(
            signal[signal.stimulus_onset],
            x="frames_id",
            y="sf",
            color="tf",
        )
        fig = go.Figure(data=lineplot.data + scatterplot.data)
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


# not in use
def get_responses_heatmap_callback(app: Dash, responses, data) -> None:
    @app.callback(
        Output("median-response-graph", "children"),
        [
            Input("demo-dropdown", "value"),
            Input("directions-checkbox", "value"),
        ],
    )
    def responses_heatmap(roi_id, dir):
        msr_for_plotting = get_median_subtracted_response(
            responses, roi_id, dir, data._sf[::-1], data._tf
        )
        x_labels, y_labels = get_labels(data, sf_inverted=True)
        fig = px.imshow(msr_for_plotting, x=x_labels, y=y_labels)

        return html.Div(
            dcc.Graph(
                id="median_response_plot",
                figure=fig,
            )
        )


# not in use
def get_symmetric_gaussian_plot_callback(app: Dash, responses, data) -> None:
    @app.callback(
        Output("gaussian-graph", "children"),
        [
            Input("demo-dropdown", "value"),
            Input("directions-checkbox", "value"),
        ],
    )
    def gaussian_plot(roi_id, dir):
        R = fit_symmetric_gaussian(
            data._sf[::-1], data._tf, responses, roi_id, data.config, dir
        )

        x_labels, y_labels = get_labels(data, sf_inverted=True)
        fig = px.imshow(R, x=x_labels, y=y_labels, aspect="equal")

        return html.Div(
            dcc.Graph(
                id="gaussian_plot",
                figure=fig,
            )
        )


def get_andermann_gaussian_plot_callback(
    app: Dash,
    median_subtracted_responses,
    downsapled_gaussians,
    oversampled_gaussians,
    fit_outputs,
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

        fig.add_trace(
            go.Heatmap(
                z=median_subtracted_responses[(roi_id, dir)],
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=1,
        )

        # Add the heatmap for the original Gaussian
        fig.add_trace(
            go.Heatmap(
                z=downsapled_gaussians[(roi_id, dir)],
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=2,
        )

        # Add the heatmap for the oversampled Gaussian
        fig.add_trace(
            go.Heatmap(
                z=oversampled_gaussians[(roi_id, dir)],
                colorscale="Viridis",
                showscale=False,
            ),
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
            width=800,
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
    app: Dash, n_roi, directions, sfs, tfs, gaussian_downsampled
) -> None:
    @app.callback(
        Output("murakami-plot", "children"),
        [
            Input("demo-dropdown", "value"),
            Input("directions-checkbox", "value"),
        ],
    )
    def murakami_plot(_roi_id, _dire):
        fig = go.Figure()

        #  range of plotly colors equal to the length of n_roi
        colors = colors = px.colors.qualitative.Light24[:n_roi]

        for roi_id in range(n_roi):
            color = colors[roi_id]
            peaks = {
                (roi_id, dire): find_peak_coordinates(
                    gaussian_downsampled[(roi_id, dire)]
                )
                # for roi_id in range(n_roi)
                for dire in directions
            }

            p = pd.DataFrame(
                {
                    "roi_id": roi_id,
                    "direction": dire,
                    "temporal_frequency": tfs[peaks[(roi_id, dire)][0]],
                    "spatial_frequency": sfs[peaks[(roi_id, dire)][1]],
                }
                for dire in directions
            )
            # print(p)

            fig.add_trace(
                go.Scatter(
                    x=p["temporal_frequency"],
                    y=p["spatial_frequency"],
                    mode="markers",
                    name=roi_id,
                    marker=dict(size=6, color=color),
                )
            )

            # # Add median point for each ROI
            median_peaks = p.groupby("roi_id").median(
                ["temporal_frequency", "spatial_frequency"]
            )
            print(f"Median peaks: {median_peaks}")

            # plot median peaks as red dots
            fig.add_trace(
                go.Scatter(
                    x=median_peaks["temporal_frequency"],
                    y=median_peaks["spatial_frequency"],
                    mode="markers",
                    marker=dict(size=12, color=color),
                    showlegend=False,
                    name="Median",
                )
            )

            if roi_id == _roi_id:
                # draw a circle around the
                # point with coordinates (roi_id, _dire)
                circle_scaling_factor_y = 0.1
                circle_scaling_factor_x = 0.1
                row = p[(p.roi_id == _roi_id) & (p.direction == _dire)].iloc[0]
                tf = row["temporal_frequency"]
                sf = row["spatial_frequency"]
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=tf - circle_scaling_factor_x * tf,
                    y0=sf - circle_scaling_factor_y * sf,
                    x1=tf + circle_scaling_factor_x * tf,
                    y1=sf + circle_scaling_factor_y * sf,
                    line_color="green",
                    line_width=2,
                    opacity=0.5,
                )

            # connect with a line all blue dots with the
            # red dot (median) one by one
            for d in directions:
                row = p[(p.roi_id == roi_id) & (p.direction == d)].iloc[0]
                tf = row["temporal_frequency"]
                sf = row["spatial_frequency"]
                fig.add_trace(
                    go.Scatter(
                        x=[tf, median_peaks["temporal_frequency"][roi_id]],
                        y=[sf, median_peaks["spatial_frequency"][roi_id]],
                        mode="lines",
                        line=dict(color=color, width=1),
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title="Murakami plot",
            yaxis_title="Spatial frequency (cycles/deg)",
            xaxis_title="Temporal frequency (Hz)",
            legend_title="ROI id",
            yaxis_type="log",
            xaxis_type="log",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            autosize=False,
            width=1200,
            height=1000,
            margin=dict(t=50, b=50, l=50, r=50),
        )

        return html.Div(
            dcc.Graph(
                id="gaussian_plot",
                figure=fig,
            )
        )
