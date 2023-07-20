import itertools
import pickle
from collections import namedtuple
from pathlib import Path

import dash
import dash_loading_spinners as dls
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import Input, Output, callback, dcc, html
from scipy.stats import pearsonr

# from rsp_vision.dashboard.pages.murakami_plot import load_data
from rsp_vision.analysis.gaussians_calculations import (
    get_gaussian_matrix_to_be_plotted,
)

dash.register_page(__name__, path="/sf_tf_facet_plot")

layout = html.Div(
    [
        dmc.Title(
            "Single-ROI visualization",
            className="page-title",
        ),
        dmc.Grid(
            children=[
                dmc.Col(
                    [
                        dmc.Text(
                            id="selected_data_str_sf_tf",
                        ),
                        dmc.Text(
                            id="selected_ROI",
                        ),
                        dmc.Text(
                            id="selected_direction",
                        ),
                        dcc.Store(id="store_choosen_roi", data={}),
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
                        ),
                        dmc.NavLink(
                            label="Single-ROI visualization",
                            href="/sf_tf_facet_plot",
                            className="navlink",
                            disabled=True,
                        ),
                        dmc.NavLink(
                            label="Polar plots",
                            href="/polar_plots",
                            className="navlink",
                        ),
                        html.Br(),
                        html.Br(),
                        dmc.Text(
                            "Choose ROI and direction. \
                            Responsive ROIs are in red.",
                        ),
                        html.Div(
                            id="roi-selection-bubble-plot",
                        ),
                        html.Div(
                            id="direction-selection-bubble-plot",
                        ),
                        dmc.Switch(
                            id="toggle-traces",
                            label="Show all traces",
                            checked=False,
                        ),
                        dmc.Text(
                            "Showing all traces could slow down plot creation",
                            size="xs",
                            color="grey",
                        ),
                    ],
                    span=2,
                ),
                dmc.Col(
                    [
                        dls.GridFade(
                            html.Div(
                                id="sf-tf-plot",
                                className="sf-tf-plot",
                            ),
                        ),
                    ],
                    span="auto",
                ),
                dmc.Col(
                    [
                        html.Div(
                            id="gaussian-graph-andermann",
                            className="gaussian-plot",
                        ),
                    ],
                    span=2,
                ),
            ],
            className="sf-tf-container",
        ),
    ],
    className="page",
)


@callback(
    Output("selected_data_str_sf_tf", "children"),
    Input("store", "data"),
)
def update_selected_data_str(store: dict) -> str:
    """This callback updates the text that shows the dataset that has been
    loaded.

    Parameters
    ----------
    store : dict
        The store contains the data that is loaded from the data table.

    Returns
    -------
    str
        The name of the dataset that has been choosen.
    """
    if store == {}:
        return "No data selected"
    else:
        return f'Dataset loaded is: {store["data"][0]}'


@callback(
    Output("roi-selection-bubble-plot", "children"),
    Input("store", "data"),
)
def roi_selection_plot(store):
    if store == {}:
        return "No data to plot"
    data = load_data(store)
    n_roi = data["n_roi"]
    responsive_rois = data["responsive_rois"]

    rois = list(range(1, n_roi + 1))
    col_n = 10
    row_n = n_roi // col_n + 1

    x = np.linspace(0, 1, col_n)
    y = np.linspace(0, 1, row_n)

    fig = go.Figure()
    for i, roi in enumerate(rois):
        fig.add_trace(
            go.Scatter(
                x=[x[i % col_n]],
                y=[-y[i // col_n]],
                mode="markers",
                marker=dict(
                    size=20,
                    color="red" if roi in responsive_rois else "gray",
                ),
                hovertemplate=f"ROI: {roi}",
            )
        )

    # annotate, add roi number
    for i, roi in enumerate(rois):
        fig.add_annotation(
            x=x[i % col_n],
            y=-y[i // col_n],
            text=str(roi),
            showarrow=False,
            font=dict(
                size=10,
                color="white",
            ),
        )

    fig.update_layout(
        width=300,
        height=30 * row_n,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return dcc.Graph(
        id="roi-selection-bubble-plot",
        figure=fig,
    )


@callback(
    Output("direction-selection-bubble-plot", "children"),
    Input("store", "data"),
)
def direction_selection_plot(store):
    if store == {}:
        return "No data to plot"
    load_data(store)
    directions = store["config"]["directions"]

    fig = px.scatter_polar(
        r=[1] * len(directions),
        theta=directions,
        range_theta=[0, 360],
        range_r=[0, 2],
        start_angle=0,
        direction="counterclockwise",
    )
    #  plot also a circle in the center
    fig.add_trace(
        go.Scatterpolar(
            r=[0],
            theta=[0],
            mode="markers",
            marker=dict(
                size=0,
                color="blue",
            ),
        )
    )

    fig.update_layout(
        width=200,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        # remove background circle
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_traces(
        marker=dict(size=30),
    )

    #  write angle on top of the circle
    for i, direction in enumerate(directions):
        fig.add_annotation(
            x=0.3 * np.cos(np.deg2rad(direction)) + 0.5,
            y=0.3 * np.sin(np.deg2rad(direction)) + 0.5,
            text=str(direction),
            showarrow=False,
            font=dict(
                size=10,
                color="white",
            ),
        )

    fig.add_annotation(
        x=0.5,
        y=0.5,
        text="all",
        showarrow=False,
        font=dict(
            size=10,
            color="white",
        ),
    )

    return dcc.Graph(
        id="direction-selection-bubble-plot",
        figure=fig,
    )


@callback(
    [
        Output("selected_ROI", "children"),
        Output("store_choosen_roi", "data"),
    ],
    #  callback on clicking on shape in heatmap - clickdata does not work
    Input("roi-selection-bubble-plot", "clickData"),
)
def update_selected_ROI(clickData):
    if clickData is None:
        default_roi_id = 9
        return "ROI 1 selected", {"roi_id": default_roi_id}
    else:
        roi_id = int(clickData["points"][0]["curveNumber"])
        return f"ROI {roi_id + 1} selected", {"roi_id": roi_id}


@callback(
    Output("selected_direction", "children"),
    Input("direction-selection-bubble-plot", "clickData"),
)
def update_selected_direction(clickData):
    if clickData is None:
        return "Pooled directions"
    else:
        if is_pooled_directions(clickData):
            return "Pooled directions"
        else:
            direction = clickData["points"][0]["theta"]
            return f"Direction {direction} selected"


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


def load_data_of_signal_dataframe(store, roi_id):
    path = (
        Path(store["path"])
        / store["subject_folder_path"]
        / store["session_folder_path"]
    )

    with open(path / f"roi_{roi_id}_signal_dataframe.pickle", "rb") as f:
        signal = pickle.load(f)

    return signal


@callback(
    Output("sf-tf-plot", "children"),
    Input("store", "data"),
    Input("store_choosen_roi", "data"),
    Input("direction-selection-bubble-plot", "clickData"),
    Input("toggle-traces", "checked"),
)
def sf_tf_grid(
    store,
    store_choosen_roi,
    direction_input,
    toggle_value,
) -> dcc.Graph:
    if store == {}:
        return "No data to plot"

    roi_id = store_choosen_roi["roi_id"]
    signal = load_data_of_signal_dataframe(store, roi_id)
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]

    if is_pooled_directions(direction_input):
        direction = "pooled"
    else:
        direction = direction_input["points"][0]["theta"]

    #  from data I need: n_days, sf_tf_combinations
    sf_tf_combinations = itertools.product(
        spatial_frequencies, temporal_frequencies
    )
    total_n_days = signal.day.max()

    Data = namedtuple("Data", ["sf_tf_combinations", "total_n_days"])
    Data(sf_tf_combinations, total_n_days)

    if direction == "pooled":
        dataframe = calculate_mean_and_median(signal)
        where_stim_is_not_na = dataframe["stimulus_repetition"].notna()

        dataframe["stimulus_repetition"][where_stim_is_not_na] = (
            dataframe["stimulus_repetition"][where_stim_is_not_na].astype(str)
            + "_"
            + dataframe["direction"][where_stim_is_not_na].astype(str)
        )
    else:
        assert isinstance(direction, int)
        signal = signal[signal.direction == direction]
        dataframe = calculate_mean_and_median(signal)

    #  remove tf and sf when they are nan
    dataframe = dataframe.dropna(subset=["tf", "sf"])

    fig = px.line(
        dataframe,
        x="stimulus_frames",
        y="signal",
        facet_col="tf",
        facet_row="sf",
        facet_col_spacing=0.005,
        facet_row_spacing=0.005,
        # width=2000,
        height=1000,
        color="stimulus_repetition",
        category_orders={
            "sf": spatial_frequencies[::-1],
            "tf": temporal_frequencies,
        },
    )

    fig.update_layout(
        title=f"SF TF traces for roi {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
        # hide y axis labels
        yaxis=dict(
            showticklabels=False,
        ),
    )

    for trace in fig.data:
        if "mean" in trace.name:
            trace.visible = True
            trace.line.color = "black"
            trace.line.width = 2
            trace.line.dash = "solid"
        elif "median" in trace.name:
            trace.visible = True
            trace.line.color = "red"
            trace.line.width = 2
            trace.line.dash = "solid"
        else:
            if not toggle_value:
                trace.visible = False
            else:
                trace.visible = True
                trace.line.width = 0.3

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
            annotation_font_size=8,
            fillcolor=color,
            opacity=0.05,
            line_width=0,
        )

    # Fake legend
    fig.add_annotation(
        x=0.01,
        y=-0.1,
        xref="paper",
        yref="paper",
        text="mean",
        showarrow=False,
        font=dict(size=15, color="black"),
    )
    fig.add_annotation(
        x=0.1,
        y=-0.7,
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


def calculate_mean_and_median(
    signal: pd.DataFrame,
) -> pd.DataFrame:
    mean_df = (
        signal.groupby(["sf", "tf", "stimulus_frames"])
        .agg({"signal": "mean"})
        .reset_index()
    )
    mean_df["stimulus_repetition"] = "mean"
    combined_df = pd.concat([signal, mean_df], ignore_index=True)

    median_df = (
        signal.groupby(["sf", "tf", "stimulus_frames"])
        .agg({"signal": "median"})
        .reset_index()
    )
    median_df["stimulus_repetition"] = "median"
    combined_df = pd.concat([combined_df, median_df], ignore_index=True)

    return combined_df


def is_pooled_directions(direction_input):
    return (direction_input is None) or (
        direction_input["points"][0]["r"] == 0
    )


@callback(
    Output("gaussian-graph-andermann", "children"),
    [
        Input("store", "data"),
        Input("store_choosen_roi", "data"),
        Input("direction-selection-bubble-plot", "clickData"),
    ],
)
def gaussian_plot(
    store: dict,
    store_choosen_roi: dict,
    direction_input: dict,
) -> html.Div:
    if store == {}:
        return "No data to plot"

    data = load_data(store)

    median_subtracted_responses = data["median_subtracted_responses"]
    fit_outputs = data["fit_outputs"]

    if store == {}:
        return "No data to plot"

    roi_id = store_choosen_roi["roi_id"]
    signal = load_data_of_signal_dataframe(store, roi_id)
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]

    if is_pooled_directions(direction_input):
        direction = "pooled"
    else:
        direction = direction_input["points"][0]["theta"]

    #  from data I need: n_days, sf_tf_combinations
    sf_tf_combinations = itertools.product(
        spatial_frequencies, temporal_frequencies
    )
    total_n_days = signal.day.max()

    Data = namedtuple("Data", ["sf_tf_combinations", "total_n_days"])
    data = Data(sf_tf_combinations, total_n_days)

    low_res_gaussian = get_6x6_gaussian(
        roi_id,
        fit_outputs,
        spatial_frequencies,
        temporal_frequencies,
        direction,
    )
    high_res_gaussian = get_custom_gaussian(
        roi_id,
        fit_outputs,
        spatial_frequencies,
        temporal_frequencies,
        direction,
    )

    # Create subplots for the two Gaussian plots
    fig = sp.make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Median subtracted response",
            "Original Gaussian",
            "Oversampled Gaussian",
        ),
    )
    uniform_sfs = uniform_tfs = np.arange(0, len(spatial_frequencies), 1)

    if isinstance(direction, int) and direction != "pooled":
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
                # z=downsampled_gaussians[(roi_id, direction)],
                z=low_res_gaussian,
                x=uniform_tfs,
                y=uniform_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=2,
            col=1,
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
                # z=oversampled_gaussians[(roi_id, direction)],
                z=high_res_gaussian,
                x=uniform_oversampled_tfs,
                y=uniform_oversampled_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=3,
            col=1,
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
            # downsampled_gaussians[(roi_id, direction)],
            low_res_gaussian,
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
                # z=downsampled_gaussians[(roi_id, "pooled")],
                z=low_res_gaussian,
                x=uniform_tfs,
                y=uniform_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=2,
            col=1,
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
                # z=oversampled_gaussians[(roi_id, "pooled")],
                z=high_res_gaussian,
                x=uniform_oversampled_tfs,
                y=uniform_oversampled_sfs,
                colorscale="Viridis",
                showscale=False,
            ),
            row=3,
            col=1,
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
            # downsampled_gaussians[(roi_id, "pooled")],
            low_res_gaussian,
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
        width=300,
        height=1000,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False,
        title_text=f"Fit Correlation: {fit_corr:.2f}, 𝜁: {fit_value:.2f}",
        #  title in the ceter and bold
        title_x=0.5,
        title_font=dict(size=20),
    )

    fig.update_xaxes(
        tickvals=uniform_tfs, ticktext=temporal_frequencies, row=1, col=1
    )
    fig.update_yaxes(
        tickvals=uniform_sfs, ticktext=spatial_frequencies, row=1, col=1
    )
    fig.update_xaxes(
        tickvals=uniform_tfs, ticktext=temporal_frequencies, row=2, col=1
    )
    fig.update_yaxes(
        tickvals=uniform_sfs, ticktext=spatial_frequencies, row=2, col=1
    )
    fig.update_yaxes(
        tickvals=uniform_oversampled_sfs[::10],
        ticktext=np.round(log_sfs[::10], 2),
        row=3,
        col=1,
    )
    fig.update_xaxes(
        tickvals=uniform_oversampled_tfs[::10],
        ticktext=np.round(log_tfs[::10], 2),
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="Spatial Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Temporal Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Spatial Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Temporal Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Spatial Frequency", row=3, col=1)
    fig.update_xaxes(title_text="Temporal Frequency", row=3, col=1)

    return html.Div(
        dcc.Graph(
            id="gaussian_plot",
            figure=fig,
        )
    )


def fit_correlation(
    gaussian: np.ndarray, median_subtracted_response: np.ndarray
) -> float:
    fit_corr, _ = pearsonr(
        median_subtracted_response.flatten(), gaussian.flatten()
    )
    return fit_corr


def get_6x6_gaussian(
    roi_id, fit_outputs, spatial_frequencies, temporal_frequencies, direction
):
    matrix = get_gaussian_matrix_to_be_plotted(
        kind="6x6 matrix",
        roi_id=roi_id,
        fit_output=fit_outputs,
        sfs=np.asarray(spatial_frequencies),
        tfs=np.asarray(temporal_frequencies),
        pooled_directions=False,
        direction=direction,
    )
    return matrix


def get_custom_gaussian(
    roi_id, fit_outputs, spatial_frequencies, temporal_frequencies, direction
):
    matrix = get_gaussian_matrix_to_be_plotted(
        kind="custom",
        roi_id=roi_id,
        fit_output=fit_outputs,
        sfs=np.asarray(spatial_frequencies),
        tfs=np.asarray(temporal_frequencies),
        pooled_directions=False,
        direction=direction,
        matrix_definition=100,
    )
    return matrix
