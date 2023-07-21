from typing import Tuple

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
from rsp_vision.dashboard.pages.helpers.data_loading import (
    load_data,
    load_data_of_signal_dataframe,
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
                            "Choose a ROI 👇 \
                            Responsive ROIs are in yellow.",
                        ),
                        dmc.Text(
                            id="selected_ROI",
                        ),
                        html.Div(
                            id="roi-selection-bubble-plot",
                        ),
                        dmc.Text("Choose a direction 👇"),
                        dmc.Text(
                            id="selected_direction",
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
    Output("roi-selection-bubble-plot", "children"),
    Input("store", "data"),
)
def roi_selection_plot(store: dict) -> dcc.Graph:
    """This callback creates the "bubble plot" that shows the ROIs and their
    responsiveness. The ROIs are represented as circles, the responsive ROIs
    are in yellow, the non-responsive ROIs are in gray.

    Parameters
    ----------
    store : dict
        The store contains the data that is loaded from the data table.

    Returns
    -------
    dcc.Graph
        The bubble plot to choose which ROI to plot.
    """
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
                    color="darkorange" if roi in responsive_rois else "gray",
                ),
                hoverinfo="none",
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
        config={"displayModeBar": False},
    )


@callback(
    Output("direction-selection-bubble-plot", "children"),
    Input("store", "data"),
)
def direction_selection_plot(store: dict) -> dcc.Graph:
    """This callback creates the plot that shows the directions that can be
    selected. The directions are represented as circles, and in the center
    there is a circle that represents all directions.

    Parameters
    ----------
    store : dict
        The store contains the data that is loaded from the data table.

    Returns
    -------
    dcc.Graph
        The bubble plot to choose which direction to plot.
    """
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
                color="royalblue",
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
        config={"displayModeBar": False},
    )


@callback(
    [
        Output("selected_ROI", "children"),
        Output("store_choosen_roi", "data"),
    ],
    Input("roi-selection-bubble-plot", "clickData"),
)
def update_selected_ROI(clickData: dict) -> Tuple[str, dict]:
    """This callback updates the text that shows the ROI that has been
    selected and the `dcc.Store` that contains the ROI id. If no ROI is
    selected, the default ROI is 0.

    Parameters
    ----------
    clickData : dict
        Which ROI has been selected in the bubble plot.

    Returns
    -------
    tuple(str, dict)
        The text that shows the ROI that has been selected and the `dcc.Store`
        that contains the ROI id.
    """
    if clickData is None:
        default_roi_id = 0
        return f"ROI {default_roi_id + 1} selected", {"roi_id": default_roi_id}
    else:
        roi_id = int(clickData["points"][0]["curveNumber"])
        return f"ROI {roi_id + 1} selected", {"roi_id": roi_id}


@callback(
    Output("selected_direction", "children"),
    Input("direction-selection-bubble-plot", "clickData"),
)
def update_selected_direction(clickData: dict) -> str:
    """This callback updates the text that shows the direction that has been
    selected. If no direction is selected, the default direction is "all".

    Parameters
    ----------
    clickData : dict
        Which direction has been selected in the bubble plot.

    Returns
    -------
    str
        The text that shows the direction that has been selected.
    """
    if clickData is None:
        return "All directions selected"
    else:
        if is_pooled_directions(clickData):
            return "All directions selected"
        else:
            direction = clickData["points"][0]["theta"]
            return f"Direction {direction} selected"


@callback(
    Output("sf-tf-plot", "children"),
    Input("store", "data"),
    Input("store_choosen_roi", "data"),
    Input("direction-selection-bubble-plot", "clickData"),
    Input("toggle-traces", "checked"),
)
def sf_tf_grid(
    store: dict,
    store_choosen_roi: dict,
    direction_input: dict,
    toggle_value: bool,
) -> dcc.Graph:
    """This callback creates the plot that shows the response of the ROI to
    the different SF-TF combinations.

    Parameters
    ----------
    store : dict
        The `dcc.Store` that contains the config information.
    store_choosen_roi : dict
        The `dcc.Store` that contains the choosen ROI id.
    direction_input : dict
        Which direction has been selected in the bubble plot.
    toggle_value : bool
        Whether to show all traces or just the mean and median.

    Returns
    -------
    dcc.Graph
        The plot that shows the response of the ROI to the different SF-TF
        combinations.
    """
    if store == {}:
        return "No data to plot"

    # Get the data
    roi_id = store_choosen_roi["roi_id"]
    signal = load_data_of_signal_dataframe(store, roi_id)
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]

    if is_pooled_directions(direction_input):
        direction = "pooled"
    else:
        direction = direction_input["points"][0]["theta"]

    # If pooled, calculate mean and median over all directions
    if direction == "pooled":
        dataframe = calculate_mean_and_median(signal)
        where_stim_is_not_na = dataframe["stimulus_repetition"].notna()

        dataframe["stimulus_repetition"][where_stim_is_not_na] = (
            dataframe["stimulus_repetition"][where_stim_is_not_na].astype(str)
            + "_"
            + dataframe["direction"][where_stim_is_not_na].astype(str)
        )
    # Else, calculate mean and median over the selected direction
    else:
        assert isinstance(direction, int)
        signal = signal[signal.direction == direction]
        dataframe = calculate_mean_and_median(signal)

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
        title=f"Response of ROI {roi_id + 1} to SF-TF combinations",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
        title_x=0.5,
        title_font=dict(size=20),
    )
    fig.for_each_yaxis(lambda y: y.update(title=""))
    fig.for_each_xaxis(lambda x: x.update(title=""))
    fig.add_annotation(
        x=-0.05,
        y=0.5,
        text="ΔF/F",
        showarrow=False,
        textangle=-90,
        xref="paper",
        yref="paper",
        font=dict(size=20),
    )
    fig.add_annotation(
        x=0.5,
        y=-0.07,
        text="Time (frames) from gray stimulus onset",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=20),
    )

    # Color the mean and median traces differently
    # If toggle_value is False, show only mean and median
    for trace in fig.data:
        if "mean" in trace.name:
            trace.visible = True
            trace.line.color = "mediumblue"
            trace.line.width = 2
            trace.line.dash = "solid"
        elif "median" in trace.name:
            trace.visible = True
            trace.line.color = "orangered"
            trace.line.width = 2
            trace.line.dash = "solid"
        else:
            if not toggle_value:
                trace.visible = False
            else:
                trace.visible = True
                trace.line.width = 0.1
                trace.line.color = "gray"

    # Add vertical rectangles to show the different stimulus phases
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

    # Write the mean and median labels
    fig.add_annotation(
        x=0.01,
        y=-0.1,
        xref="paper",
        yref="paper",
        text="MEAN",
        showarrow=False,
        font=dict(size=20, color="mediumblue"),
    )
    fig.add_annotation(
        x=0.07,
        y=-0.1,
        xref="paper",
        yref="paper",
        text="MEDIAN",
        showarrow=False,
        font=dict(size=20, color="orangered"),
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
    """This method calculates the mean and median of the signal dataframe
    over the different stimulus repetitions.

    Parameters
    ----------
    signal : pd.DataFrame
        The signal dataframe containing the data of the ROI. If the direction
        is pooled, the dataframe contains the data of all directions and the
        mean and median are calculated over all directions. If the direction
        is not pooled, the dataframe contains the data of the selected
        direction and the mean and median are calculated over it.
        These computations are done in the `sf_tf_grid` callback.
    Returns
    -------
    pd.DataFrame
        The signal dataframe containing the mean and median of the signal
        appended to the original dataframe at the end.
    """
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


def is_pooled_directions(direction_input: dict) -> bool:
    """This method checks whether the direction is pooled or not.

    Parameters
    ----------
    direction_input : dict
        Which direction has been selected in the bubble plot.

    Returns
    -------
    bool
        Whether the direction is pooled or not.
    """
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
    """This callback creates the heatmaps that show the Gaussian fits of the
    ROI to the different SF-TF combinations.
    The first heatmap shows the median subtracted response of the ROI to the
    different SF-TF combinations. These values are the ones that are fitted
    in the analysis pipeline. The fitting parameters are used to compute two
    Gaussian fits: one with the original SF-TF combinations and one with
    oversampled SF-TF combinations. These two Gaussian fits are shown in the
    second and third heatmap, respectively.

    Parameters
    ----------
    store : dict
        The `dcc.Store` that contains the config information.
    store_choosen_roi : dict
        The `dcc.Store` that contains the choosen ROI id.
    direction_input : dict
        Which direction has been selected in the bubble plot.

    Returns
    -------
    html.Div
        The heatmaps that show the Gaussian fits of the ROI to the different
        SF-TF combinations.
        The title also shows the correlation between the median subtracted
        response and the Gaussian fit and the 𝜁 value of the fit, which is a
        measure of the "speed tuning" of the ROI.
    """
    if store == {}:
        return "No data to plot"

    # Get the data
    data = load_data(store)

    median_subtracted_responses = data["median_subtracted_responses"]
    fit_outputs = data["fit_outputs"]

    if store == {}:
        return "No data to plot"

    roi_id = store_choosen_roi["roi_id"]
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]

    if is_pooled_directions(direction_input):
        direction = "pooled"
    else:
        direction = direction_input["points"][0]["theta"]

    # Get the Gaussian fits from the fit outputs
    # First get the Gaussian fits for the original SF-TF combinations
    # If the direction is pooled, it automatically takes the pooled direction
    low_res_gaussian = get_6x6_gaussian(
        roi_id,
        fit_outputs,
        spatial_frequencies,
        temporal_frequencies,
        direction,
    )
    # Then get the Gaussian fits for the oversampled SF-TF combinations
    # I am creating a 100x100 matrix
    matrix_definition = store["config"]["fitting"]["oversampling_factor"]
    high_res_gaussian = get_custom_gaussian(
        roi_id,
        fit_outputs,
        spatial_frequencies,
        temporal_frequencies,
        direction,
        matrix_definition=matrix_definition,
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

    # A linear array to set the axis of the first two heatmaps
    uniform_sfs = uniform_tfs = np.arange(0, len(spatial_frequencies), 1)

    #  Add the heatmap for the median subtracted response
    fig.add_trace(
        go.Heatmap(
            z=median_subtracted_responses[(roi_id, direction)]
            if not is_pooled_directions(direction_input)
            else median_subtracted_responses[(roi_id, "pooled")],
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
            z=low_res_gaussian,
            x=uniform_tfs,
            y=uniform_sfs,
            colorscale="Viridis",
            showscale=False,
        ),
        row=2,
        col=1,
    )

    # Linear arrays to set the axis of the third heatmap
    uniform_oversampled_sfs = np.linspace(
        0, matrix_definition - 1, matrix_definition
    )
    uniform_oversampled_tfs = np.linspace(
        0, matrix_definition - 1, matrix_definition
    )

    # Add the heatmap for the oversampled Gaussian
    fig.add_trace(
        go.Heatmap(
            z=high_res_gaussian,
            x=uniform_oversampled_tfs,
            y=uniform_oversampled_sfs,
            colorscale="Viridis",
            showscale=True,
        ),
        row=3,
        col=1,
    )

    #  Calculate the correlation between the median subtracted response and 𝜻
    fit_corr = fit_correlation(
        low_res_gaussian,
        median_subtracted_responses[(roi_id, "pooled")]
        if is_pooled_directions(direction_input)
        else median_subtracted_responses[(roi_id, direction)],
    )
    fit_value = (
        fit_outputs[(roi_id, direction)][-1]
        if not is_pooled_directions(direction_input)
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
        title_x=0.5,
        title_font=dict(size=20),
        title=dict(
            y=1,
            pad=dict(b=30),
        ),
    )
    #  set zmax and zmin to have the same color scale for all heatmaps
    fig.update_traces(
        zmax=100,
        zmin=0,
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

    longer_array_sfs = np.linspace(
        min(spatial_frequencies),
        max(spatial_frequencies),
        num=matrix_definition,
    )

    longer_array_tfs = np.linspace(
        min(temporal_frequencies),
        max(temporal_frequencies),
        num=matrix_definition,
    )
    fig.update_yaxes(
        tickvals=uniform_oversampled_sfs[::10],
        ticktext=np.round(longer_array_sfs[::10], 2),
        row=3,
        col=1,
    )
    fig.update_xaxes(
        tickvals=uniform_oversampled_tfs[::10],
        ticktext=np.round(longer_array_tfs[::10], 2),
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
    """This method calculates the correlation between the median subtracted
    response and the Gaussian fit (6x6 matrix).

    Parameters
    ----------
    gaussian : np.ndarray
        The Gaussian fit (6x6 matrix).
    median_subtracted_response : np.ndarray
        The median subtracted response.

    Returns
    -------
    float
        The correlation between the median subtracted response and the Gaussian
        fit (6x6 matrix).
    """
    fit_corr, _ = pearsonr(
        median_subtracted_response.flatten(), gaussian.flatten()
    )
    return fit_corr


def get_6x6_gaussian(
    roi_id: int,
    fit_outputs: dict,
    spatial_frequencies: list,
    temporal_frequencies: list,
    direction: str,
) -> np.ndarray:
    """This method gets the Gaussian fit (6x6 matrix) from the fit outputs.

    Parameters
    ----------
    roi_id : int
        The choosen ROI id.
    fit_outputs : dict
        The fit outputs previously calculated.
    spatial_frequencies : list
        The spatial frequencies as from the config file.
    temporal_frequencies : list
        The temporal frequencies as from the config file.
    direction : str
        The choosen direction. It can be "pooled" or an integer.

    Returns
    -------
    np.ndarray
        The Gaussian fit (6x6 matrix).
    """
    matrix = get_gaussian_matrix_to_be_plotted(
        kind="6x6 matrix",
        roi_id=roi_id,
        fit_output=fit_outputs,
        sfs=np.asarray(spatial_frequencies),
        tfs=np.asarray(temporal_frequencies),
        direction=direction,
    )
    return matrix


def get_custom_gaussian(
    roi_id: int,
    fit_outputs: dict,
    spatial_frequencies: list,
    temporal_frequencies: list,
    direction: str,
    matrix_definition: int = 100,
) -> np.ndarray:
    """This method gets the Gaussian fit (NxN matrix) from the fit outputs.
    N is the matrix definition, which is 100 by default.

    Parameters
    ----------
    roi_id : int
        The choosen ROI id.
    fit_outputs : dict
        The fit outputs previously calculated.
    spatial_frequencies : list
        The spatial frequencies as from the config file.
    temporal_frequencies : list
        The temporal frequencies as from the config file.
    direction : str
        The choosen direction. It can be "pooled" or an integer.
    matrix_definition : int, optional
        The matrix definition, by default 100.

    Returns
    -------
    np.ndarray
        The Gaussian fit (NxN matrix).
    """
    matrix = get_gaussian_matrix_to_be_plotted(
        kind="custom",
        roi_id=roi_id,
        fit_output=fit_outputs,
        sfs=np.asarray(spatial_frequencies),
        tfs=np.asarray(temporal_frequencies),
        direction=direction,
        matrix_definition=matrix_definition,
    )
    return matrix
