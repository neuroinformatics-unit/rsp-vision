import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import Dash, Input, Output, dcc, html

from rsp_vision.dashboard.query_dataframes import (
    fit_andermann_gaussian,
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


def get_andermann_gaussian_plot_callback(app: Dash, responses, data) -> None:
    @app.callback(
        Output("gaussian-graph-andermann", "children"),
        [
            Input("demo-dropdown", "value"),
            Input("directions-checkbox", "value"),
        ],
    )
    def gaussian_plot(roi_id, dir):
        sfs_inverted = data._sf  # [::-1]
        tfs = data._tf
        (
            R,
            R_oversampled,
            fit_corr,
            𝜻_power_law_exp,
            oversampled_sfs_inverted,
            oversampled_tfs,
        ) = fit_andermann_gaussian(
            sfs_inverted, tfs, responses, roi_id, data.config, dir
        )

        x_labels, y_labels = get_labels(data, sf_inverted=True)

        msr_for_plotting = get_median_subtracted_response(
            responses, roi_id, dir, data._sf, data._tf
        )

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
                z=msr_for_plotting,
                # x=sfs_inverted,
                # y=tfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=1,
        )

        # Add the heatmap for the original Gaussian
        fig.add_trace(
            go.Heatmap(
                z=R,
                # x=sfs_inverted,
                # y=tfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=2,
        )

        # Add the heatmap for the oversampled Gaussian
        fig.add_trace(
            go.Heatmap(
                z=R_oversampled,
                # x=oversampled_sfs_inverted,
                # y=oversampled_tfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=3,
        )

        # Update layout to maintain the aspect ratio
        fig.update_layout(
            autosize=False,
            width=800,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=False,
            title_text=f"Fit Correlation: {fit_corr:.2f}, \
                𝜁: {𝜻_power_law_exp:.2f}",
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
