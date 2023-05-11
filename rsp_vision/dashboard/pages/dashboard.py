import pickle

import dash
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
from dash import dcc, html

from rsp_vision.dashboard.callbacks.direction_controller import (
    get_update_circle_figure_callback,
    get_update_radio_items_callback,
)
from rsp_vision.dashboard.callbacks.gaussian_plots import (
    get_andermann_gaussian_plot_callback,
)
from rsp_vision.dashboard.callbacks.murakami_plot import (
    get_murakami_plot_callback,
)
from rsp_vision.dashboard.callbacks.plotting_helpers import (
    get_df_sf_tf_combo_plot,
)
from rsp_vision.dashboard.callbacks.polar_plots import (
    get_polar_plot_callback,
    get_polar_plot_facet_callback,
)
from rsp_vision.dashboard.callbacks.session_plot import (
    get_update_fig_all_sessions_callback,
)
from rsp_vision.dashboard.callbacks.sf_tf_facet_plot import (
    get_sf_tf_grid_callback,
)
from rsp_vision.dashboard.layout import get_sidebar
from rsp_vision.objects.photon_data import PhotonData

dash.register_page(
    __name__,
    path="/dashboard/",
    # redirect_from=["/home/"],
    # args=["selected_dataset"],
)


def layout(data=None):
    with open(f"/Users/laura/data/output/{data}.pickle", "rb") as f:
        data: PhotonData = pickle.load(f)

    signal = data.signal
    responsive_rois = data.responsive_rois
    n_roi = data.n_roi
    rois = list(range(data.n_roi))
    directions = data.directions
    spatial_frequencies = data.spatial_frequencies
    temporal_frequencies = data.temporal_frequencies
    downsampled_gaussians = data.downsampled_gaussian
    oversampled_gaussians = data.oversampled_gaussian
    fit_outputs = data.fit_output
    median_subtracted_responses = data.median_subtracted_response
    counts = get_df_sf_tf_combo_plot(signal, data)

    sidebar = get_sidebar(responsive_rois, rois, directions)

    layout = html.Div(
        [
            dbc.Container(
                [
                    html.Div([dcc.Location(id="url"), sidebar]),
                    dls.GridFade(
                        html.Div(
                            id="session-graph",
                        ),
                    ),
                    dls.GridFade(
                        html.Div(
                            id="sf_tf-graph",
                        ),
                    ),
                    dls.GridFade(
                        html.Div(
                            id="gaussian-graph-andermann",
                        ),
                    ),
                    dls.GridFade(
                        html.Div(
                            id="murakami-plot",
                        ),
                    ),
                    dls.GridFade(
                        html.Div(
                            id="polar-plot",
                        ),
                    ),
                    dls.GridFade(
                        html.Div(
                            id="polar-plot-facet",
                        ),
                    ),
                ],
                fluid=True,
            ),
            # get_update_fig_all_sessions_callback,
            # get_andermann_gaussian_plot_callback,
            # get_murakami_plot_callback,
            # get_polar_plot_callback,
            # get_polar_plot_facet_callback,
            # get_update_radio_items_callback,
            # get_update_circle_figure_callback,
            # get_sf_tf_grid_callback,
        ],
        id="main-container",
    )

    get_update_fig_all_sessions_callback(signal)
    get_sf_tf_grid_callback(signal, data, counts)
    get_andermann_gaussian_plot_callback(
        median_subtracted_responses=median_subtracted_responses,
        downsampled_gaussians=downsampled_gaussians,
        oversampled_gaussians=oversampled_gaussians,
        fit_outputs=fit_outputs,
        spatial_frequencies=spatial_frequencies,
        temporal_frequencies=temporal_frequencies,
    )
    get_murakami_plot_callback(
        n_roi=n_roi,
        directions=directions,
        spatial_frequencies=spatial_frequencies,
        temporal_frequencies=temporal_frequencies,
        oversampled_gaussians=oversampled_gaussians,
        responsive_rois=responsive_rois,
        config=data.config,
    )
    get_polar_plot_callback(
        directions=directions,
        spatial_frequencies=spatial_frequencies,
        temporal_frequencies=temporal_frequencies,
        downsampled_gaussians=downsampled_gaussians,
        median_subtracted_responses=median_subtracted_responses,
    )
    get_polar_plot_facet_callback(
        directions=directions,
        spatial_frequencies=spatial_frequencies,
        temporal_frequencies=temporal_frequencies,
        downsampled_gaussians=downsampled_gaussians,
        median_subtracted_responses=median_subtracted_responses,
    )
    get_update_radio_items_callback()
    get_update_circle_figure_callback(directions)

    return layout
