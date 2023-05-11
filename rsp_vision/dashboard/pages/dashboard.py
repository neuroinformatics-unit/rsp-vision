import pickle
from plotly.subplots import make_subplots
import itertools
import plotly.express as px

import dash
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
from dash import dcc, html, Input, Output, callback, State
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from rsp_vision.dashboard.callbacks.plotting_helpers import (
    fit_correlation,
    get_peaks_dataframe,
    find_peak_coordinates,
    get_circle_coordinates,
    get_direction_plot_for_controller,
    get_df_sf_tf_combo_plot,
    get_dataframe_for_facet_plot,
    get_dataframe_for_facet_plot_pooled_directions,
)

from rsp_vision.dashboard.layout import get_sidebar
from rsp_vision.objects.photon_data import PhotonData

dash.register_page(
    __name__,
    path="/dashboard/",
)


def layout(data=None):
    with open(f"/Users/laura/data/output/{data}.pickle", "rb") as f:
        data: PhotonData = pickle.load(f)

    global data_ext
    data_ext = data

    global signal
    global responsive_rois
    global n_roi
    global rois
    global directions
    global spatial_frequencies
    global temporal_frequencies
    global downsampled_gaussians
    global oversampled_gaussians
    global fit_outputs
    global median_subtracted_responses

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

    global counts
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
                    # dls.GridFade(
                    #     html.Div(
                    #         id="test-div",
                    #     ),
                    # ),
                ],
                fluid=True,
            ),
        ],
        id="main-container",
    )

    return layout


# @callback(
#     Output(component_id='test-div', component_property='children'),
#     Input(component_id='test-div', component_property='children')
# )
# def update_output_div(useless_value):
#     if (data_ext is None):
#         print("unexpected...")
#         return 'data loading failed'
#     print (data_ext.day_stim)
#     return data_ext.day_stim


@callback(
    [
        Output("direction-store", "data"),
        Output("selected-direction", "children"),
    ],
    Input("directions-circle", "clickData"),
    State("direction-store", "data"),
)
def update_radio_items(clickData: dict, current_data: dict) -> tuple:
    if clickData is not None:
        clicked_point = clickData["points"][0]
        if "customdata" in clicked_point:
            direction = clicked_point["customdata"]
        elif "text" in clicked_point and clicked_point["text"] == "ALL":
            direction = "all"
        else:
            return current_data, current_data["value"]

        return {"value": direction}, direction
    else:
        return current_data, current_data["value"]


@callback(
    Output("directions-circle", "figure"),
    Input("selected-direction", "children"),
)
def update_circle_figure(selected_direction: int) -> go.Figure:
    circle_x, circle_y = get_circle_coordinates(directions)
    return get_direction_plot_for_controller(
        directions, circle_x, circle_y, selected_direction
    )


@callback(
    Output("gaussian-graph-andermann", "children"),
    [
        Input("roi-choice-dropdown", "value"),
        Input("direction-store", "data"),
    ],
)
def gaussian_plot(roi_id: int, direction_input: dict) -> html.Div:
    direction = direction_input["value"]

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
    uniform_sfs = uniform_tfs = np.arange(0, len(spatial_frequencies), 1)

    if isinstance(direction, int) and direction != "all":
        #  Add the heatmap for the median subtracted response
        fig.add_trace(
            go.Heatmap(
                z=median_subtracted_responses[(roi_id, direction)],
                x=uniform_tfs,
                y=uniform_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=1,
        )

        # Add the heatmap for the original Gaussian
        fig.add_trace(
            go.Heatmap(
                z=downsampled_gaussians[(roi_id, direction)],
                x=uniform_tfs,
                y=uniform_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=2,
        )

        # Add the heatmap for the oversampled Gaussian
        # I tried with the log plot, it does not look good
        oversampling_factor = 100
        uniform_oversampled_sfs = np.linspace(
            0, oversampling_factor - 1, oversampling_factor
        )
        uniform_oversampled_tfs = np.linspace(
            0, oversampling_factor - 1, oversampling_factor
        )

        fig.add_trace(
            go.Heatmap(
                z=oversampled_gaussians[(roi_id, direction)],
                x=uniform_oversampled_tfs,
                y=uniform_oversampled_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=3,
        )

        log_sfs = np.linspace(
            min(spatial_frequencies),
            max(spatial_frequencies),
            num=oversampling_factor,
        )

        log_tfs = np.linspace(
            min(temporal_frequencies),
            max(temporal_frequencies),
            num=oversampling_factor,
        )

        fit_corr = fit_correlation(
            downsampled_gaussians[(roi_id, direction)],
            median_subtracted_responses[(roi_id, direction)],
        )
    else:
        #  Add the heatmap for the median subtracted response
        fig.add_trace(
            go.Heatmap(
                z=median_subtracted_responses[(roi_id, "pooled")],
                x=uniform_tfs,
                y=uniform_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=1,
        )

        # Add the heatmap for the original Gaussian
        fig.add_trace(
            go.Heatmap(
                z=downsampled_gaussians[(roi_id, "pooled")],
                x=uniform_tfs,
                y=uniform_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=2,
        )

        # Add the heatmap for the oversampled Gaussian
        oversampling_factor = 100
        uniform_oversampled_sfs = np.linspace(
            0, oversampling_factor - 1, oversampling_factor
        )
        uniform_oversampled_tfs = np.linspace(
            0, oversampling_factor - 1, oversampling_factor
        )

        fig.add_trace(
            go.Heatmap(
                z=oversampled_gaussians[(roi_id, "pooled")],
                x=uniform_oversampled_tfs,
                y=uniform_oversampled_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1,
            col=3,
        )

        log_sfs = np.linspace(
            min(spatial_frequencies),
            max(spatial_frequencies),
            num=oversampling_factor,
        )

        log_tfs = np.linspace(
            min(temporal_frequencies),
            max(temporal_frequencies),
            num=oversampling_factor,
        )

        fit_corr = fit_correlation(
            downsampled_gaussians[(roi_id, "pooled")],
            median_subtracted_responses[(roi_id, "pooled")],
        )

    fit_value = (
        fit_outputs[(roi_id, direction)][-1]
        if isinstance(direction, int) and direction != "all"
        else fit_outputs[(roi_id, "pooled")][-1]
    )

    # Update layout to maintain the aspect ratio
    fig.update_layout(
        autosize=False,
        width=1100,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False,
        title_text=f"Fit Correlation: {fit_corr:.2f}, 𝜁: {fit_value:.2f}",
    )

    fig.update_xaxes(
        tickvals=uniform_tfs, ticktext=temporal_frequencies, row=1, col=1
    )
    fig.update_yaxes(
        tickvals=uniform_sfs, ticktext=spatial_frequencies, row=1, col=1
    )
    fig.update_xaxes(
        tickvals=uniform_tfs, ticktext=temporal_frequencies, row=1, col=2
    )
    fig.update_yaxes(
        tickvals=uniform_sfs, ticktext=spatial_frequencies, row=1, col=2
    )
    fig.update_yaxes(
        tickvals=uniform_oversampled_sfs[::10],
        ticktext=np.round(log_sfs[::10], 2),
        row=1,
        col=3,
    )
    fig.update_xaxes(
        tickvals=uniform_oversampled_tfs[::10],
        ticktext=np.round(log_tfs[::10], 2),
        row=1,
        col=3,
    )

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


@callback(
    Output("murakami-plot", "children"),
    [
        Input("roi-choice-dropdown", "value"),
        Input("direction-store", "data"),
        Input("which-roi-to-show-in-murakami-plot", "value"),
        Input("murakami-plot-scale", "value"),
    ],
)
def murakami_plot(
    roi_id_input: int, direction_input: dict, rois_to_show: str, scale: str
) -> html.Div:
    direction_input = direction_input["value"]

    # colors = matplotlib.colors.cnames.values()

    fig = go.Figure()

    def simplified_murakami_plot(roi_id):
        # color = colors[roi_id]
        peaks = {
            roi_id: find_peak_coordinates(
                oversampled_gaussians[(roi_id, "pooled")],
                spatial_frequencies,
                temporal_frequencies,
                data_ext.config,
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

    def figure_for_murakami_plot(roi_id):
        # color = colors[roi_id]
        peaks = {
            (roi_id, dire): find_peak_coordinates(
                oversampled_gaussians[(roi_id, dire)],
                spatial_frequencies,
                temporal_frequencies,
                data_ext.config,
            )
            for dire in directions
        }

        p = pd.DataFrame(
            {
                "roi_id": roi_id,
                "direction": dire,
                "temporal_frequency": peaks[(roi_id, dire)][0],
                "spatial_frequency": peaks[(roi_id, dire)][1],
            }
            for dire in directions
        )

        median_peaks = p.groupby("roi_id").median(
            ["temporal_frequency", "spatial_frequency"]
        )

        for i, d in enumerate(directions):
            row = p[(p.roi_id == roi_id) & (p.direction == d)].iloc[0]
            tf = row["temporal_frequency"]
            sf = row["spatial_frequency"]
            fig.add_trace(
                go.Scatter(
                    x=[tf, median_peaks["temporal_frequency"][roi_id]],
                    y=[sf, median_peaks["spatial_frequency"][roi_id]],
                    mode="markers",
                    marker=dict(
                        # color=color,
                        size=10
                    ),
                    name=f"ROI {roi_id + 1}" if i == 0 else "",
                    showlegend=True if i == 0 else False,
                    marker_line_width=2 if roi_id in responsive_rois else 0,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[tf, median_peaks["temporal_frequency"][roi_id]],
                    y=[sf, median_peaks["spatial_frequency"][roi_id]],
                    mode="lines",
                    line=dict(
                        # color=color,
                        width=1
                    ),
                    showlegend=False,
                )
            )

        if roi_id == roi_id_input:
            row = p[
                (p.roi_id == roi_id_input) & (p.direction == direction_input)
            ].iloc[0]
            tf = row["temporal_frequency"]
            sf = row["spatial_frequency"]

            fig.add_trace(
                go.Scatter(
                    x=[tf],
                    y=[sf],
                    mode="markers",
                    marker=dict(color="red", size=20),
                    showlegend=False,
                )
            )

        return fig

    if rois_to_show == "choosen":
        if isinstance(direction_input, int) and direction_input != "all":
            fig = figure_for_murakami_plot(roi_id_input)
        elif direction_input == "all":
            simplified_murakami_plot(roi_id_input)
    elif rois_to_show == "responsive":
        for roi_id in responsive_rois:
            if isinstance(direction_input, int) and direction_input != "all":
                fig = figure_for_murakami_plot(roi_id)
            elif direction_input == "all":
                simplified_murakami_plot(roi_id)
    elif rois_to_show == "all":
        for roi_id in range(n_roi):
            if isinstance(direction_input, int) and direction_input != "all":
                fig = figure_for_murakami_plot(roi_id)
            elif direction_input == "all":
                simplified_murakami_plot(roi_id)

    fig.update_layout(
        title=f"Murakami plot, direction {direction_input}",
        yaxis_title="Spatial frequency (cycles/deg)",
        xaxis_title="Temporal frequency (Hz)",
        legend_title="ROI",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        autosize=False,
        width=1200,
        height=1000,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    if scale == "log":
        fig.update_yaxes(type="log")
        fig.update_xaxes(type="log")
        fig.update_xaxes(
            # range=[0, 1],
            title_text="Temporal frequency (Hz)",
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            # range=[0, 0.1],
            title_text="Spatial frequency (cycles/deg)",
            showgrid=False,
            zeroline=False,
        )
    else:
        fig.update_yaxes(type="linear")
        fig.update_xaxes(type="linear")
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
    for i in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]:
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
    for i in [0.5, 1, 2, 4, 8, 16]:
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

    return html.Div(
        dcc.Graph(
            id="gaussian_plot",
            figure=fig,
        )
    )


@callback(
    Output("polar-plot", "children"),
    [
        Input("roi-choice-dropdown", "value"),
        Input("polar-plot-gaussian-or-original", "value"),
        Input("polar-plot-mean-or-median-or-cumulative", "value"),
    ],
)
def polar_plot(
    roi_id: int,
    gaussian_or_original: str,
    mean_or_median_or_cumulative: str,
) -> html.Div:
    p_sorted = get_peaks_dataframe(
        gaussian_or_original,
        roi_id,
        directions,
        spatial_frequencies,
        temporal_frequencies,
        median_subtracted_responses,
        downsampled_gaussians,
    )

    pivot_table = p_sorted.pivot(
        index=["temporal_frequency", "spatial_frequency"],
        columns="direction",
        values="corresponding_value",
    )

    subplot_titles = [
        "Responses across sf/tf for each direction",
        "",
    ]
    if mean_or_median_or_cumulative == "mean":
        total_values = pivot_table.mean(axis=0)
        subplot_titles[1] = "mean responses"
    elif mean_or_median_or_cumulative == "median":
        total_values = pivot_table.median(axis=0)
        subplot_titles[1] = "median responses"
    elif mean_or_median_or_cumulative == "cumulative":
        pivot_table = pivot_table - pivot_table.min().min()
        pivot_table = pivot_table / pivot_table.max().max()
        total_values = pivot_table.sum(axis=0)
        subplot_titles[1] = "cumulative responses"

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=subplot_titles,
    )
    colors = px.colors.qualitative.Light24[: int(len(spatial_frequencies))]

    for i, (tf, sf) in enumerate(
        itertools.product(temporal_frequencies, spatial_frequencies)
    ):
        row = p_sorted[
            (p_sorted.temporal_frequency == tf)
            & (p_sorted.spatial_frequency == sf)
        ]
        color = colors[i % len(spatial_frequencies)]

        r_values = row["corresponding_value"].tolist()
        theta_values = row["direction"].tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=r_values + [r_values[0]],
                theta=theta_values + [theta_values[0]],
                mode="lines",
                thetaunit="degrees",
                line=dict(color=color, width=4),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatterpolar(
            r=total_values.tolist() + [total_values.tolist()[0]],
            theta=total_values.index.tolist()
            + [total_values.index.tolist()[0]],
            mode="lines",
            thetaunit="degrees",
            name="Cumulative",
            line=dict(color="black", width=3),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        plot_bgcolor="rgbaDash, (0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        width=1600,
        height=800,
        title=dict(
            text="Analysis of responses across sf/tf for each direction",
            font=dict(size=24, color="black", family="Arial, bold"),
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=1,
        ),
        annotations=[
            dict(
                font=dict(size=18, color="black", family="Arial, bold"),
                xanchor="center",
                yanchor="top",
                x=0.22,
                y=1.08,
                showarrow=False,
            ),
            dict(
                font=dict(size=18, color="black", family="Arial, bold"),
                xanchor="center",
                yanchor="top",
                x=0.78,
                y=1.08,
                showarrow=False,
            ),
        ],
    )

    return html.Div(
        dcc.Graph(
            id="polar_plot",
            figure=fig,
        )
    )


@callback(
    Output("polar-plot-facet", "children"),
    [
        Input("roi-choice-dropdown", "value"),
        Input("polar-plot-gaussian-or-original", "value"),
    ],
)
def polar_plot_facet(roi_id: int, gaussian_or_original: str) -> html.Div:
    sorted_sfs = sorted(spatial_frequencies, reverse=True)
    sorted_tfs = sorted(temporal_frequencies)

    p_sorted = get_peaks_dataframe(
        gaussian_or_original=gaussian_or_original,
        roi_id=roi_id,
        directions=directions,
        spatial_frequencies=sorted_sfs,
        temporal_frequencies=sorted_tfs,
        median_subtracted_responses=median_subtracted_responses,
        downsampled_gaussians=downsampled_gaussians,
    )

    max_value = p_sorted["corresponding_value"].max()

    ncols = len(sorted_sfs)
    nrows = len(sorted_tfs)

    subplot_titles = [
        f"sf: {sf}, tf: {tf}"
        for sf, tf in itertools.product(sorted_sfs, sorted_tfs)
    ]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "polar"}] * ncols] * nrows,
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    for sf_idx, tf_idx in itertools.product(
        range(len(sorted_sfs)), range(len(sorted_tfs))
    ):
        tf = sorted_tfs[tf_idx]
        sf = sorted_sfs[(sf_idx + 1) * -1]

        subset = p_sorted[
            (p_sorted["temporal_frequency"] == tf)
            & (p_sorted["spatial_frequency"] == sf)
        ]

        r_values = subset["corresponding_value"].tolist()
        theta_values = subset["direction"].tolist()

        fig.add_trace(
            go.Scatterpolar(
                r=r_values + [r_values[0]],
                theta=theta_values + [theta_values[0]],
                mode="lines",
                fill="toself",
                fillcolor="rgba(0, 0, 0, 0.2)",
                marker=dict(size=10),
                line=dict(width=1),
                thetaunit="degrees",
                showlegend=False,
                subplot=f"polar{tf_idx * ncols + sf_idx + 1}",
            ),
            row=sf_idx + 1,
            col=tf_idx + 1,
        )

        subplot_name = f"polar{tf_idx * ncols + sf_idx + 1}"
        fig.update_layout(
            {
                subplot_name: dict(
                    radialaxis=dict(
                        visible=False,
                        range=[0, max_value],
                        gridcolor="rgba(0, 0, 0, 0)",
                    ),
                    angularaxis=dict(
                        visible=False,
                        showticklabels=False,
                        showgrid=False,
                        gridcolor="rgba(0, 0, 0, 0)",
                    ),
                )
            }
        )

    fig.update_layout(
        title=f"SF TF polar plot for roi {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        width=1500,
        height=1500,
    )

    return html.Div(
        dcc.Graph(
            id="polar-plot-facet-internal",
            figure=fig,
        )
    )


@callback(
    Output("session-graph", "children"),
    Input("roi-choice-dropdown", "value"),
)
def update_fig_all_sessions(roi_id: int) -> dcc.Graph:
    pastel_colors = [
        "#acd0f8",
        "#ace7d0",
    ]

    unique_session_ids = signal["session_id"].unique()
    line_data = []

    for i, session_id in enumerate(unique_session_ids):
        session_data = signal[
            (signal.roi_id == roi_id) & (signal.session_id == session_id)
        ]
        line_data.append(
            go.Scatter(
                x=session_data["frames_id"],
                y=session_data["signal"],
                mode="lines",
                line=dict(
                    color=pastel_colors[i % len(pastel_colors)], width=0.3
                ),
                showlegend=False,
            )
        )

    responses = signal[signal.stimulus_onset]
    scatterplot = px.scatter(
        responses,
        x="frames_id",
        y="sf",
    )

    scatterplot.update_traces(showlegend=False)

    fig = go.Figure(data=line_data + list(scatterplot.data))
    fig.update_layout(
        title=f"Signal across sessions, roi: {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
        width=1500,
    )

    return html.Div(
        dcc.Graph(
            id="my-graph",
            figure=fig,
        )
    )


@callback(
    Output("sf_tf-graph", "children"),
    [
        Input("roi-choice-dropdown", "value"),
        Input("direction-store", "data"),
        Input("toggle-traces", "value"),
    ],
)
def sf_tf_grid(
    roi_id: int, direction_input: dict, toggle_value: str
) -> dcc.Graph:
    direction = direction_input["value"]
    if direction == "all":
        dataframe = get_dataframe_for_facet_plot_pooled_directions(
            signal, roi_id
        )
    else:
        assert isinstance(direction, int)
        dataframe = get_dataframe_for_facet_plot(
            signal, data_ext, counts, roi_id, direction
        )

    fig = px.line(
        dataframe,
        x="stimulus_frames",
        y="signal",
        facet_col="tf",
        facet_row="sf",
        facet_col_spacing=0.005,
        facet_row_spacing=0.005,
        width=2000,
        height=1200,
        color="signal_kind",
        category_orders={
            "sf": data_ext.spatial_frequencies[::-1],
            "tf": data_ext.temporal_frequencies,
        },
    )

    fig.update_layout(
        title=f"SF TF traces for roi {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
    )
    for trace in fig.data:
        if "mean" in trace.name:
            trace.line.color = "black"
            trace.line.width = 3
            trace.line.dash = "solid"
        elif "median" in trace.name:
            trace.line.color = "red"
            trace.line.width = 3
            trace.line.dash = "solid"
        else:
            if "ALL" not in toggle_value:
                trace.visible = False
            else:
                trace.line.width = 0.5

    for x0, x1, text, color in [
        (0, 75, "gray", "green"),
        (75, 150, "static", "pink"),
        (150, 225, "drift", "blue"),
    ]:
        fig.add_vrect(
            x0=x0,
            x1=x1,
            row="all",
            col="all",
            annotation_text=text,
            annotation_position="top left",
            annotation_font_size=15,
            fillcolor=color,
            opacity=0.1,
            line_width=0,
        )

    # Fake legend
    fig.add_annotation(
        x=0.9,
        y=0.97,
        xref="paper",
        yref="paper",
        text="mean",
        showarrow=False,
        font=dict(size=15, color="black"),
    )
    fig.add_annotation(
        x=0.95,
        y=0.97,
        xref="paper",
        yref="paper",
        text="median",
        showarrow=False,
        font=dict(size=15, color="red"),
    )

    return html.Div(
        dcc.Graph(
            id="sf_tf_plot",
            figure=fig,
        )
    )
