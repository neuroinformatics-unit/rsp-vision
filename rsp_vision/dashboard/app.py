import pickle

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from rsp_vision.dashboard.callbacks import (
    get_gaussian_plot_callback,
    get_responses_heatmap_callback,
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
    responses = data.responses
    # p_values = analysis.p_values
    # magnitude = analysis.magintude_over_medians
    responsive_rois = data.responsive_rois

    # config = data.config

    rois = list(range(data.n_roi))
    directions = list(data._dir)
    # sfs = np.sort(data.uniques["sf"])
    # sfs_inverted = np.sort(data.uniques["sf"])[::-1]
    # tfs = np.sort(data.uniques["tf"])

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
            html.Div(
                id="median-response-graph",
            ),
            html.Div(
                id="gaussian-graph",
            ),
        ]
    )

    # CALLBACKS
    # =============================================================================
    get_update_fig_all_sessions_callback(app, signal)
    get_sf_tf_grid_callback(app, signal, data, counts)
    get_responses_heatmap_callback(app, responses, data)
    get_gaussian_plot_callback(app, responses, data)

    return app
