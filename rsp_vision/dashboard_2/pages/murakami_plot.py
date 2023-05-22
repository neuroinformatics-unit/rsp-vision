import pickle
from pathlib import Path

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

dash.register_page(__name__, path="/murakami_plot")

layout = html.Div(
    [
        dmc.Title(
            "Murakami",
            order=2,
            className="page-title",
        ),
        dmc.Grid(
            children=[
                dmc.Col(
                    [
                        dmc.Switch(
                            id="show-only-responsive",
                            label="Show only responsive ROIs",
                            checked=True,
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
    Output("murakami-plot", "children"),
    [
        Input("store", "data"),
        Input("show-only-responsive", "checked"),
    ],
)
def murakami_plot(store, show_only_responsive):
    if store == {}:
        return "No data to plot"
    else:
        oversampling_factor = store["oversampling_factor"]
        spatial_frequencies = store["spatial_frequencies"]
        temporal_frequencies = store["temporal_frequencies"]
        data = load_data(store)
        responsive_rois = data["responsive_rois"]
        n_roi = data["n_roi"]
        oversampled_gaussians = data["oversampled_gaussians"]

        total_roi = (
            responsive_rois if show_only_responsive else list(range(n_roi))
        )
        print(f"total roi: {total_roi}")

        fig = go.Figure()

        for roi_id in total_roi:
            print(f"plotting roi {roi_id}")
            fig = simplified_murakami_plot(
                roi_id=roi_id,
                fig=fig,
                oversampling_factor=oversampling_factor,
                responsive_rois=responsive_rois,
                oversampled_gaussians=oversampled_gaussians[
                    (roi_id, "pooled")
                ],
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

        return dcc.Graph(
            id="gaussian_plot",
            figure=fig,
        )


def simplified_murakami_plot(
    roi_id,
    fig,
    oversampling_factor,
    responsive_rois,
    oversampled_gaussians,
    spatial_frequencies,
    temporal_frequencies,
):
    # color = colors[roi_id]
    print("finding peaks...")
    peaks = {
        roi_id: find_peak_coordinates(
            oversampled_gaussian=oversampled_gaussians,
            spatial_frequencies=np.asarray(spatial_frequencies),
            temporal_frequencies=np.asarray(temporal_frequencies),
            oversampling_factor=oversampling_factor,
        )
    }
    print("peaks found")

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
    sub = store["data"][0]
    ses = store["data"][1]
    line = store["data"][2]
    id = store["data"][3]
    hemisphere = store["data"][4]
    brain_region = store["data"][5]
    monitor_position = store["data"][6]
    subject_folder = f"sub-{sub:03d}_line-{line}_id-{id}"
    session_folder = (
        f"ses-{ses:03d}_hemisphere-{hemisphere}"
        + +f"_region-{brain_region}_monitor-{monitor_position}"
    )

    path = (
        Path(store["path"])
        / subject_folder
        / session_folder
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
    print("finding peak coordinates")
    # find the peak indices
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
