import pickle
from pathlib import Path

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from rsp_vision.analysis.gaussians_calculations import (
    get_gaussian_matrix_to_be_plotted,
)

dash.register_page(__name__, path="/murakami_plot")

layout = html.Div(
    [
        dmc.Title(
            "Murakami Plot",
            className="page-title",
        ),
        dmc.Grid(
            children=[
                dmc.Col(
                    [
                        dmc.Text(
                            id="selected_data_str_murakami",
                        ),
                        html.Br(),
                        dmc.NavLink(
                            label="Back to Data Table",
                            href="/",
                            className="navlink",
                        ),
                        dmc.NavLink(
                            label="Murakami plot",
                            href="/murakami_plot",
                            className="navlink",
                            disabled=True,
                        ),
                        dmc.NavLink(
                            label="SF-TF facet plot and gaussians",
                            href="/sf_tf_facet_plot",
                            className="navlink",
                        ),
                        dmc.NavLink(
                            label="Polar plots",
                            href="/polar_plots",
                            className="navlink",
                        ),
                        html.Br(),
                        dmc.Switch(
                            id="show-only-responsive",
                            label="Show only responsive ROIs",
                            checked=True,
                            className="responsive-switch",
                        ),
                        html.Br(),
                        dmc.Text(
                            "Responsive ROIs are shown in red, "
                            + "non-responsive ROIs are shown in black.",
                            size="xs",
                            color="grey",
                            className="responsive-switch-text",
                        ),
                        dmc.Alert(
                            "No responsive ROIs found",
                            id="responsive-rois-warnings",
                            title="Warning",
                            color="yellow",
                            hide=True,
                        ),
                    ],
                    span=2,
                ),
                dmc.Col(
                    [
                        html.Div(
                            id="murakami-plot",
                            className="murakami-plot",
                        ),
                    ],
                    span="auto",
                    offset=1,
                ),
            ],
            className="murakami-container",
        ),
    ],
    className="page",
)


@callback(
    Output("responsive-rois-warnings", "hide"),
    Input("store", "data"),
)
def responsive_rois_warnings(store):
    if store == {}:
        return True
    else:
        data = load_data(store)
        responsive_rois = data["responsive_rois"]
        print(responsive_rois)
        if (responsive_rois == 0) | (responsive_rois == set()):
            return False
        else:
            return True


@callback(
    Output("selected_data_str_murakami", "children"),
    Input("store", "data"),
)
def update_selected_data_str(store):
    if store == {}:
        return "No data selected"
    else:
        return f'Dataset loaded is: {store["data"][0]}'


@callback(
    Output("murakami-plot", "children"),
    [
        Input("store", "data"),
        Input("show-only-responsive", "checked"),
    ],
)
def murakami_plot(store, show_only_responsive):
    if store == {}:
        return "No data to plot"

    data = load_data(store)

    # prepare data
    responsive_rois = data["responsive_rois"]
    n_roi = data["n_roi"]
    matrix_definition = 100
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]
    fit_outputs = data["fit_outputs"]
    fitted_gaussian_matrix = call_get_gaussian_matrix_to_be_plotted(
        n_roi,
        fit_outputs,
        spatial_frequencies,
        temporal_frequencies,
        matrix_definition,
    )

    total_roi = responsive_rois if show_only_responsive else list(range(n_roi))

    # plot
    fig = go.Figure()
    fig = add_data_in_figure(
        all_roi=total_roi,
        fig=fig,
        matrix_definition=matrix_definition,
        responsive_rois=responsive_rois,
        fitted_gaussian_matrix=fitted_gaussian_matrix,
        spatial_frequencies=spatial_frequencies,
        temporal_frequencies=temporal_frequencies,
    )
    fig = prettify_murakami_plot(
        fig, spatial_frequencies, temporal_frequencies
    )

    return dcc.Graph(figure=fig)


def prettify_murakami_plot(fig, spatial_frequencies, temporal_frequencies):
    fig.update_layout(
        yaxis_title="Spatial frequency (cycles/deg)",
        xaxis_title="Temporal frequency (Hz)",
        legend_title="ROI",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        autosize=False,
        width=600,
        height=600,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    fig.update_xaxes(
        range=[-0.05, 17],
        title_text="Temporal frequency (Hz)",
        showgrid=False,
        zeroline=False,
        tickvals=[],
    )
    fig.update_yaxes(
        range=[0, 0.33],
        title_text="Spatial frequency (cycles/deg)",
        showgrid=False,
        zeroline=False,
        tickvals=[],
    )

    #  draw horizontal lines
    for i in spatial_frequencies:
        fig.add_shape(
            type="line",
            x0=0.25,
            y0=i,
            x1=16.1,
            y1=i,
            line=dict(color="Grey", width=1),
        )
        #  add annotations for horizontal lines
        fig.add_annotation(
            x=0.05,
            y=i,
            text=f"{i}",
            showarrow=False,
            yshift=0,
            xshift=-10,
            font=dict(color="Black"),
        )

    #  draw vertical lines
    for i in temporal_frequencies:
        fig.add_shape(
            type="line",
            x0=i,
            y0=0.001,
            x1=i,
            y1=0.33,
            line=dict(color="Grey", width=1),
        )
        #  add annotations for vertical lines
        fig.add_annotation(
            x=i,
            y=0.001,
            text=f"{i}",
            showarrow=False,
            yshift=510,
            xshift=0,
            font=dict(color="Black"),
        )
    return fig


def add_data_in_figure(
    all_roi,
    fig,
    matrix_definition,
    responsive_rois,
    fitted_gaussian_matrix,
    spatial_frequencies,
    temporal_frequencies,
):
    peaks = {
        roi_id: find_peak_coordinates(
            fitted_gaussian_matrix=fitted_gaussian_matrix[(roi_id, "pooled")],
            spatial_frequencies=np.asarray(spatial_frequencies),
            temporal_frequencies=np.asarray(temporal_frequencies),
            matrix_definition=matrix_definition,
        )
        for roi_id in all_roi
    }

    p = pd.DataFrame(
        {
            "roi_id": roi_id,
            "temporal_frequency": peaks[roi_id][0],
            "spatial_frequency": peaks[roi_id][1],
        }
        for roi_id in all_roi
    )

    median_peaks = p.median()

    #  dots for ROIs
    for roi_id in all_roi:
        row = p[(p.roi_id == roi_id)].iloc[0]
        tf = row["temporal_frequency"]
        sf = row["spatial_frequency"]
        fig.add_trace(
            go.Scatter(
                x=[tf, median_peaks["temporal_frequency"]],
                y=[sf, median_peaks["spatial_frequency"]],
                mode="markers",
                marker=dict(
                    color="red" if roi_id in responsive_rois else "black",
                    size=10,
                ),
                name=f"ROI {roi_id + 1}",
                showlegend=False,
            )
        )

        # lines to connect to the median dot
        fig.add_trace(
            go.Scatter(
                x=[tf, median_peaks["temporal_frequency"]],
                y=[sf, median_peaks["spatial_frequency"]],
                mode="lines",
                line=dict(color="Grey", width=1),
                showlegend=False,
            )
        )

    return fig


def load_data(store):
    path = (
        Path(store["path"])
        / store["subject_folder_path"]
        / store["session_folder_path"]
        / "gaussians_fits_and_roi_info.pickle"
    )
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def find_peak_coordinates(
    fitted_gaussian_matrix: np.ndarray,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    matrix_definition: int,
):
    peak_indices = np.unravel_index(
        np.argmax(fitted_gaussian_matrix), fitted_gaussian_matrix.shape
    )

    spatial_freq_linspace = np.linspace(
        spatial_frequencies.min(),
        spatial_frequencies.max(),
        matrix_definition,
    )
    temporal_freq_linspace = np.linspace(
        temporal_frequencies.min(),
        temporal_frequencies.max(),
        matrix_definition,
    )

    sf = spatial_freq_linspace[peak_indices[0]]
    tf = temporal_freq_linspace[peak_indices[1]]
    return tf, sf


def call_get_gaussian_matrix_to_be_plotted(
    n_roi: int,
    fit_outputs: dict,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    matrix_definition: int,
):
    fitted_gaussian_matrix = {}

    for roi_id in range(n_roi):
        fitted_gaussian_matrix[
            (roi_id, "pooled")
        ] = get_gaussian_matrix_to_be_plotted(
            kind="custom",
            roi_id=roi_id,
            fit_output=fit_outputs,
            sfs=np.asarray(spatial_frequencies),
            tfs=np.asarray(temporal_frequencies),
            pooled_directions=True,
            matrix_definition=matrix_definition,
        )

    return fitted_gaussian_matrix
