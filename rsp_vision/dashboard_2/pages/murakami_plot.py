import pickle
from pathlib import Path

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from rsp_vision.analysis.gaussians_calculations import get_gaussian_matrix_to_be_plotted

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
                        dmc.Switch(
                            id="show-only-responsive",
                            label="Show only responsive ROIs",
                            checked=True,
                        ),
                        html.Br(),
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
                ),
            ],
            className="murakami-container",
        ),
    ],
    className="page",
)


@callback(
    Output("selected_data_str_murakami", "children"),
    Input("store", "data"),
)
def update_selected_data_str(store):
    if store == {}:
        return "No data selected"
    else:
        return f'Dataset loaded is: {store["data"][2]}'


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

    responsive_rois = data["responsive_rois"]
    n_roi = data["n_roi"]
    # oversampled_gaussians = data["oversampled_gaussians"]
    oversampling_factor = store["config"]["fitting"]["oversampling_factor"]
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]

    

    oversampled_gaussians = call_get_gaussian_matrix_to_be_plotted(
        n_roi,
        data,
        spatial_frequencies,
        temporal_frequencies,
        oversampling_factor,
    )

    print(f"n_roi: {n_roi}")

    total_roi = responsive_rois if show_only_responsive else list(range(n_roi))

    fig = go.Figure()

    for roi_id in total_roi:
        fig = simplified_murakami_plot(
            roi_id=roi_id,
            fig=fig,
            oversampling_factor=oversampling_factor,
            responsive_rois=responsive_rois,
            oversampled_gaussians=oversampled_gaussians[(roi_id, "pooled")],
            spatial_frequencies=spatial_frequencies,
            temporal_frequencies=temporal_frequencies,
        )

    fig.update_layout(
        title="Murakami plot",
        yaxis_title="Spatial frequency (cycles/deg)",
        xaxis_title="Temporal frequency (Hz)",
        legend_title="ROI",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        # paper_bgcolor="rgba(0, 0, 0, 0)",
        autosize=False,
        width=1000,
        height=800,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    fig.update_xaxes(
        range=[-0.05, 16.1],
        title_text="Temporal frequency (Hz)",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        range=[0, 0.33],
        title_text="Spatial frequency (cycles/deg)",
        showgrid=False,
        zeroline=False,
    )
    #  draw horizontal lines
    for i in spatial_frequencies:
        fig.add_shape(
            type="line",
            x0=0.25,
            y0=i,
            x1=16.1,
            y1=i,
            line=dict(color="LightGray", width=1),
        )
        #  add annotations for horizontal lines
        fig.add_annotation(
            x=0.05,
            y=i,
            text=f"{i}",
            showarrow=False,
            yshift=10,
            xshift=10,
            font=dict(color="LightGray"),
        )

    for i in temporal_frequencies:
        fig.add_shape(
            type="line",
            x0=i,
            y0=0.001,
            x1=i,
            y1=0.33,
            line=dict(color="LightGray", width=1),
        )
        #  add annotations for vertical lines
        fig.add_annotation(
            x=i,
            y=0.001,
            text=f"{i}",
            showarrow=False,
            yshift=10,
            xshift=10,
            font=dict(color="LightGray"),
        )

    return dcc.Graph(id="gaussian_plot", figure=fig)


def simplified_murakami_plot(
    roi_id,
    fig,
    oversampling_factor,
    responsive_rois,
    oversampled_gaussians,
    spatial_frequencies,
    temporal_frequencies,
):
    peaks = {
        roi_id: find_peak_coordinates(
            oversampled_gaussian=oversampled_gaussians,
            spatial_frequencies=np.asarray(spatial_frequencies),
            temporal_frequencies=np.asarray(temporal_frequencies),
            oversampling_factor=oversampling_factor,
        )
    }

    p = pd.DataFrame(
        {
            "roi_id": roi_id,
            "temporal_frequency": peaks[roi_id][0],
            "spatial_frequency": peaks[roi_id][1],
        },
        index=[0],
    )

    median_peaks = p.groupby("roi_id").median(
        ["temporal_frequency", "spatial_frequency"]
    )

    row = p[(p.roi_id == roi_id)].iloc[0]
    tf = row["temporal_frequency"]
    sf = row["spatial_frequency"]
    fig.add_trace(
        go.Scatter(
            x=[tf, median_peaks["temporal_frequency"][roi_id]],
            y=[sf, median_peaks["spatial_frequency"][roi_id]],
            mode="markers",
            marker=dict(
                # color=color,
                size=20
            ),
            showlegend=True,
            name=f"ROI {roi_id}",
            marker_line_width=2 if roi_id in responsive_rois else 0,
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
    oversampled_gaussian: np.ndarray,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    oversampling_factor: int,
):
    peak_indices = np.unravel_index(
        np.argmax(oversampled_gaussian), oversampled_gaussian.shape
    )

    spatial_freq_linspace = np.linspace(
        spatial_frequencies.min(),
        spatial_frequencies.max(),
        oversampling_factor,
    )
    temporal_freq_linspace = np.linspace(
        temporal_frequencies.min(),
        temporal_frequencies.max(),
        oversampling_factor,
    )

    sf = spatial_freq_linspace[peak_indices[0]]
    tf = temporal_freq_linspace[peak_indices[1]]
    return tf, sf


def call_get_gaussian_matrix_to_be_plotted(
        n_roi: int,
        data: dict,
        spatial_frequencies: np.ndarray,
        temporal_frequencies: np.ndarray,
        oversampling_factor: int,
):
    oversampled_gaussians = {}

    
    for roi_id in range(n_roi):
        print((roi_id, "pooled"))
        print(data["fit_outputs"][(roi_id, "pooled")])
        oversampled_gaussians[(roi_id, "pooled")] = get_gaussian_matrix_to_be_plotted(
            kind="custom",
            roi_id=roi_id,
            fit_output=data["fit_outputs"],
            sfs= np.asarray(spatial_frequencies),
            tfs= np.asarray(temporal_frequencies),
            pooled_directions=True,
            matrix_definition=oversampling_factor
        )

    return oversampled_gaussians
        

