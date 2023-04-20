import pickle

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from rsp_vision.dashboard.callbacks import (
    get_andermann_gaussian_plot_callback,
    get_murakami_plot_callback,
    get_sf_tf_grid_callback,
    get_update_fig_all_sessions_callback,
)
from rsp_vision.dashboard.layout import get_sidebar
from rsp_vision.dashboard.query_dataframes import get_df_sf_tf_combo_plot


def get_app():
    app = Dash(
        __name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP]
    )

    # LOAD DATA
    # =============================================================================
    with open("AK_1111739_hL_RSPd_monitor_front_data.pickle", "rb") as f:
        data = pickle.load(f)

    # Unpack data
    signal = data.signal
    responsive_rois = data.responsive_rois
    n_roi = data.n_roi
    rois = list(range(data.n_roi))
    directions = list(data._dir)
    sfs = data._sf
    tfs = data._tf

    downsapled_gaussians = data.downsampled_gaussian
    oversampled_gaussians = data.oversampled_gaussian
    fit_outputs = data.fit_output
    median_subtracted_responses = data.median_subtracted_response

    # ADAPT DATA FRAMES
    # =============================================================================
    counts = get_df_sf_tf_combo_plot(signal, data)

    # LAYOUT
    # =============================================================================

    sidebar = get_sidebar(responsive_rois, rois, directions)

    app.layout = dbc.Container(
        [
            html.Div([dcc.Location(id="url"), sidebar]),
            html.Div(
                id="example-graph",
            ),
            html.Div(
                id="sf_tf-graph",
            ),
            # html.Div(
            #     id="median-response-graph",
            # ),
            # html.Div(
            #     id="gaussian-graph",
            # ),
            html.Div(
                id="gaussian-graph-andermann",
            ),
            html.Div(
                id="murakami-plot",
            ),
        ]
    )

    # CALLBACKS
    # =============================================================================
    get_update_fig_all_sessions_callback(app, signal)
    get_sf_tf_grid_callback(app, signal, data, counts)
    # get_responses_heatmap_callback(app, responses, data)
    # get_symmetric_gaussian_plot_callback(app, responses, data)
    get_andermann_gaussian_plot_callback(
        app,
        median_subtracted_responses,
        downsapled_gaussians,
        oversampled_gaussians,
        fit_outputs,
    )
    get_murakami_plot_callback(
        app,
        n_roi,
        directions,
        sfs,
        tfs,
        oversampled_gaussians,
        responsive_rois,
    )

    return app
