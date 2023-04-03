import pickle

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

from rsp_vision.dashboard.callbacks import (
    get_andermann_gaussian_plot_callback,
    get_murakami_plot_callback,
    get_polar_plot_callback,
    get_polar_plot_facet_callback,
    get_sf_tf_grid_callback,
    get_update_circle_figure_callback,
    get_update_fig_all_sessions_callback,
    get_update_radio_items_callback,
)
from rsp_vision.dashboard.layout import get_sidebar
from rsp_vision.dashboard.plotting_helpers import get_df_sf_tf_combo_plot


def get_app() -> Dash:
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
    directions = [int(x) for x in list(data._dir)]
    sfs = data._sf
    tfs = data._tf

    downsampled_gaussians = data.downsampled_gaussian
    oversampled_gaussians = data.oversampled_gaussian
    fit_outputs = data.fit_output
    median_subtracted_responses = data.median_subtracted_response

    # ADAPT DATA FRAMES
    # =============================================================================
    counts = get_df_sf_tf_combo_plot(signal, data)

    # CALLBACKS
    # =============================================================================
    get_update_fig_all_sessions_callback(app, signal)
    get_sf_tf_grid_callback(app, signal, data, counts)
    get_andermann_gaussian_plot_callback(
        app=app,
        median_subtracted_responses=median_subtracted_responses,
        downsampled_gaussians=downsampled_gaussians,
        oversampled_gaussians=oversampled_gaussians,
        fit_outputs=fit_outputs,
        sfs=sfs,
        tfs=tfs,
    )
    get_murakami_plot_callback(
        app=app,
        n_roi=n_roi,
        directions=directions,
        sfs=sfs,
        tfs=tfs,
        oversampled_gaussians=oversampled_gaussians,
        responsive_rois=responsive_rois,
    )
    get_polar_plot_callback(
        app=app,
        directions=directions,
        sfs=sfs,
        tfs=tfs,
        downsampled_gaussians=downsampled_gaussians,
        median_subtracted_responses=median_subtracted_responses,
    )
    get_polar_plot_facet_callback(
        app=app,
        directions=directions,
        sfs=sfs,
        tfs=tfs,
        downsampled_gaussians=downsampled_gaussians,
        median_subtracted_responses=median_subtracted_responses,
    )
    get_update_radio_items_callback(app)
    get_update_circle_figure_callback(app, directions)

    # LAYOUT
    # =============================================================================
    sidebar = get_sidebar(responsive_rois, rois, directions)

    app.layout = dbc.Container(
        [
            html.Div([dcc.Location(id="url"), sidebar]),
            html.Div(
                id="session-graph",
            ),
            html.Div(
                id="sf_tf-graph",
            ),
            html.Div(
                id="gaussian-graph-andermann",
            ),
            html.Div(
                id="murakami-plot",
            ),
            html.Div(
                id="polar-plot",
            ),
            html.Div(
                id="polar-plot-facet",
            ),
        ]
    )

    return app
