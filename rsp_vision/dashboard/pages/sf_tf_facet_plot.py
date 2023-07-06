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
            "SF-TF facet plot and gaussians",
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
                        dcc.Store(id="store_choosen_roi_and_dir", data={}),
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
                        ),
                        dmc.NavLink(
                            label="SF-TF facet plot and gaussians",
                            href="/sf_tf_facet_plot",
                            className="navlink",
                            disabled=True,
                        ),
                        dmc.NavLink(
                            label="Polar plots",
                            href="/polar_plots",
                            className="navlink",
                        ),
                        dmc.Text(
                            "Choose your ROI, the responsive ones are in red",
                        ),
                        html.Div(
                            id="roi_and_direction_selection_bubble_plot",
                            className="roi-and-direction-selection-bubble-plot",
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
                    span=3,
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
    Output("roi_and_direction_selection_bubble_plot", "children"),
    Input("store", "data"),
)
def rois_and_direction_plot(store):
    if store == {}:
        return "No data to plot"
    data = load_data(store)
    n_roi = data["n_roi"]
    responsive_rois = data["responsive_rois"]

    # make radius list
    #  for the first 36 rois, radius = 1, then it grows by 1 for every 36 rois
    angle_diff = 30
    circle = int(360 / angle_diff)
    radius = [1] * n_roi
    for i in range(1, n_roi // circle + 1):
        radius[i * circle :] = [i + 1] * (n_roi - i * circle)

    fig = px.scatter_polar(
        r=radius,
        theta=range(0, angle_diff * n_roi, angle_diff),
        range_theta=[0, 360],
        range_r=[0, (n_roi // circle + 1) + 1],
        start_angle=0,
        direction="counterclockwise",
        color=[i in responsive_rois for i in range(n_roi)],
        color_discrete_sequence=["gray", "red"],
        hover_name=[str(i) for i in range(n_roi)],
        size=[3] * n_roi,
    )

    fig.update_layout(
        width=200,
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(visible=False),
        ),
        showlegend=False,
    )

    return dcc.Graph(
        id="roi_and_direction_selection_bubble_plot",
        figure=fig,
    )


@callback(
    [
        Output("selected_ROI", "children"),
        Output("store_choosen_roi_and_dir", "data"),
    ],
    #  callback on clicking on shape in heatmap - clickdata does not work
    Input("roi_and_direction_selection_bubble_plot", "clickData"),
)
def update_selected_ROI(clickData):
    if clickData is None:
        default_roi_id = 0
        return "ROI 1 selected", {"roi_id": default_roi_id}
    else:
        roi_id = int(clickData["points"][0]["hovertext"])
        return f"ROI {roi_id + 1} selected", {"roi_id": roi_id}


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
    # [
    Input("store_choosen_roi_and_dir", "data"),
    #     # Input("direction-store", "data"),
    #     # Input("toggle-traces", "value"),
    # ],
)
def sf_tf_grid(
    store,
    store_choosen_roi_and_dir,
) -> dcc.Graph:
    if store == {}:
        return "No data to plot"

    roi_id = store_choosen_roi_and_dir["roi_id"]
    signal = load_data_of_signal_dataframe(store, roi_id)
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]

    # direction = direction_input["value"]
    direction = 90

    store["config"]["n_frames_per_session"]

    #  from data I need: n_days, sf_tf_combinations
    sf_tf_combinations = itertools.product(
        spatial_frequencies, temporal_frequencies
    )
    total_n_days = signal.day.max()

    Data = namedtuple("Data", ["sf_tf_combinations", "total_n_days"])
    Data(sf_tf_combinations, total_n_days)

    if direction == "all":
        dataframe = get_dataframe_for_facet_plot_pooled_directions(signal)
    else:
        assert isinstance(direction, int)
        dataframe = get_dataframe_for_facet_plot(signal, direction)

    fig = px.line(
        dataframe,
        x="stimulus_frames",
        y="signal",
        facet_col="tf",
        facet_row="sf",
        facet_col_spacing=0.005,
        facet_row_spacing=0.005,
        # width=2000,
        height=800,
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
    )
    for trace in fig.data:
        if "mean" in trace.name:
            trace.line.color = "black"
            trace.line.width = 1
            trace.line.dash = "solid"
        elif "median" in trace.name:
            trace.line.color = "red"
            trace.line.width = 1
            trace.line.dash = "solid"
        else:
            # if "ALL" not in toggle_value:
            #     trace.visible = False
            # else:
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


def get_dataframe_for_facet_plot_pooled_directions(
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


def get_dataframe_for_facet_plot(
    signal: pd.DataFrame,
    direction: int,
) -> pd.DataFrame:
    signal = signal[signal.direction == direction]
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


@callback(
    Output("gaussian-graph-andermann", "children"),
    [
        Input("store", "data"),
        Input("store_choosen_roi_and_dir", "data"),
        #     Input("direction-store", "data"),
    ],
)
def gaussian_plot(
    store: dict,
    store_choosen_roi_and_dir: dict,
) -> html.Div:
    if store == {}:
        return "No data to plot"

    data = load_data(store)

    median_subtracted_responses = data["median_subtracted_responses"]
    fit_outputs = data["fit_outputs"]

    if store == {}:
        return "No data to plot"

    roi_id = store_choosen_roi_and_dir["roi_id"]
    signal = load_data_of_signal_dataframe(store, roi_id)
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]

    # direction = direction_input["value"]
    direction = 90

    #  from data I need: n_days, sf_tf_combinations
    sf_tf_combinations = itertools.product(
        spatial_frequencies, temporal_frequencies
    )
    total_n_days = signal.day.max()

    Data = namedtuple("Data", ["sf_tf_combinations", "total_n_days"])
    data = Data(sf_tf_combinations, total_n_days)

    # direction = direction_input["value"]
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
        width=400,
        height=1000,
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
