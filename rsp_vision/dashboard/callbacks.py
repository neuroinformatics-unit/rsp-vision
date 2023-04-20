import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from rsp_vision.dashboard.query_dataframes import (
    fit_elliptical_gaussian,
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
            facet_col="sf",
            facet_row="tf",
            width=1500,
            height=800,
            color="signal_kind",
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


def get_gaussian_plot_callback(app: Dash, responses, data) -> None:
    @app.callback(
        Output("gaussian-graph", "children"),
        [
            Input("demo-dropdown", "value"),
            Input("directions-checkbox", "value"),
        ],
    )
    def gaussian_plot(roi_id, dir):
        R = fit_elliptical_gaussian(
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


def get_labels(data, sf_inverted=True):
    if sf_inverted:
        y_labels = list(map(str, data._sf[::-1].tolist()))
    else:
        y_labels = list(map(str, data._sf.tolist()))
    x_labels = list(map(str, data._tf.tolist()))

    return x_labels, y_labels
