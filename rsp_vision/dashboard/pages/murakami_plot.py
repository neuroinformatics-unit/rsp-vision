import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html, register_page

from rsp_vision.dashboard.pages.helpers.calculations_for_plotting import (
    find_peak_coordinates,
    fit_correlation,
    get_gaussian_matrix_to_be_plotted_for_all_rois,
)
from rsp_vision.dashboard.pages.helpers.data_loading import load_data

register_page(__name__, path="/murakami_plot")

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
                            label="Single-ROI visualization",
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
                        dmc.Switch(
                            id="include-bad-fit",
                            label="Include bad fits",
                            checked=False,
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
                    dmc.Center(
                        html.Div(
                            id="murakami-plot",
                            className="murakami-plot",
                        ),
                    ),
                    span="auto",
                    offset=1,
                ),
                dmc.Col(
                    dmc.Center(
                        html.Div(
                            id="murakami-plot2",
                            className="murakami-plot",
                        ),
                    ),
                    span="auto",
                    offset=1,
                ),
                dmc.Col(
                    html.Div(
                        id="speed-tuning-plot",
                        className="speed-tuning-plot",
                    ),
                    span="auto",
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
def responsive_rois_warnings(store: dict) -> bool:
    """This callback hides the warning message if there are responsive ROIs,
    and shows it if there are not. By default, the warning message is hidden.
    Parameters
    ----------
    store : dict
        The store contains the data that is loaded from the data table.

    Returns
    -------
    bool
        Whether to hide the warning message or not.
    """
    if store == {}:
        return True
    else:
        data = load_data(store)
        responsive_rois = data["responsive_neurons"]
        if (responsive_rois == 0) | (responsive_rois == set()):
            return False
        else:
            return True


@callback(
    Output("selected_data_str_murakami", "children"),
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
    Output("murakami-plot", "children"),
    [
        Input("store", "data"),
        Input("show-only-responsive", "checked"),
    ],
)
def murakami_plot(store: dict, show_only_responsive: bool) -> dcc.Graph:
    """This callback generates the Murakami plot. It is called Murakami plot
    because it is inspired by the visualization on Murakami et al. (2017).
    Peak responses for each roi are shown as a "scatter plot" in a 2D space,
    where the x-axis is the spatial frequency and the y-axis is the temporal
    frequency. The dots are connected by lines to the median dot creating
    a start shape.
    The peak responses represent the spatial and temporal frequency that
    maximizes the response of the roi.

    Parameters
    ----------
    store : dict
        The store contains the data that is loaded from the data table.
    show_only_responsive : bool
        Whether to show only the responsive ROIs or not.

    Returns
    -------
    dcc.Graph
        The Murakami plot.
    """
    if store == {}:
        return "No data to plot"

    data = load_data(store)

    # prepare data
    responsive_rois = data["responsive_neurons"]
    data["n_neurons"]
    neurons_idx = data["idx_neurons"]
    matrix_dimension = 100
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]
    fit_outputs = data["fit_outputs"]
    fitted_gaussian_matrix = get_gaussian_matrix_to_be_plotted_for_all_rois(
        neurons_idx,
        fit_outputs,
        spatial_frequencies,
        temporal_frequencies,
        matrix_dimension,
    )

    total_roi = (
        responsive_rois if show_only_responsive else data["idx_neurons"]
    )

    # plot
    fig = go.Figure()
    fig = add_data_in_figure(
        all_roi=total_roi,
        fig=fig,
        matrix_dimension=matrix_dimension,
        responsive_rois=responsive_rois,
        fitted_gaussian_matrix=fitted_gaussian_matrix,
        spatial_frequencies=spatial_frequencies,
        temporal_frequencies=temporal_frequencies,
    )
    fig = prettify_murakami_plot(
        fig, spatial_frequencies, temporal_frequencies
    )

    return dcc.Graph(figure=fig)


def prettify_murakami_plot(
    fig: go.Figure,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
) -> go.Figure:
    """This method takes a figure and edits its aesthetics by manipulating
    the properties of the figure. It is specific for the Murakami plot.

    Parameters
    ----------
    fig : go.Figure
        The figure to be edited.
    spatial_frequencies : np.ndarray
        The spatial frequencies that are used in the experiment.
    temporal_frequencies : np.ndarray
        The temporal frequencies that are used in the experiment.

    Returns
    -------
    go.Figure
        The edited figure.
    """
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
    all_roi: list,
    fig: go.Figure,
    matrix_dimension: int,
    fitted_gaussian_matrix: pd.DataFrame,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    connected_lines: bool = True,
    responsive_rois: list = [],
    marker_style: str = "responsiveness",
    exponential_coeficient: list = [],
) -> go.Figure:
    """For each roi, this method adds a dot in the Murakami plot (representing
    the peak response of the roi) and the lines connecting the dots to the
    median dot.

    Parameters
    ----------
    all_roi : list
        The list of all the ROIs.
    fig : go.Figure
        The figure to which the data is to be added.
    matrix_dimension : int
        The matrix definition used in the experiment. It specifies
        the precision of the sf/tf peaks that are to be found. Needs to match
        the matrix definition used to generate the fitted_gaussian_matrix.
    responsive_rois : list
        The list of responsive ROIs.
    fitted_gaussian_matrix : pd.DataFrame
        The fitted gaussian matrix obtained from the precalculated fits.
    spatial_frequencies : np.ndarray
        The spatial frequencies that are used in the experiment.
    temporal_frequencies : np.ndarray
        The temporal frequencies that are used in the experiment.

    Returns
    -------
    go.Figure
        The figure with the data added.
    """
    peaks = {
        roi_id: find_peak_coordinates(
            fitted_gaussian_matrix=fitted_gaussian_matrix[(roi_id, "pooled")],
            spatial_frequencies=np.asarray(spatial_frequencies),
            temporal_frequencies=np.asarray(temporal_frequencies),
            matrix_dimension=matrix_dimension,
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
    if marker_style == "responsiveness":
        for roi_id in all_roi:
            row = p[(p.roi_id == roi_id)].iloc[0]
            tf = row["temporal_frequency"]
            sf = row["spatial_frequency"]

            marker = dict(
                color="red" if roi_id in responsive_rois else "black",
                size=10,
            )
            fig.add_trace(
                go.Scatter(
                    x=[tf, median_peaks["temporal_frequency"]],
                    y=[sf, median_peaks["spatial_frequency"]],
                    mode="markers",
                    marker=marker,
                    name=f"ROI {roi_id + 1}",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[tf, median_peaks["temporal_frequency"]],
                    y=[sf, median_peaks["spatial_frequency"]],
                    mode="lines",
                    line=dict(color="Grey", width=1),
                    showlegend=False,
                )
            )
    elif marker_style == "gaussian_fit":
        #  just do a scatter plot with the gaussian fit values
        fig.add_trace(
            go.Scatter(
                x=p["temporal_frequency"],
                y=p["spatial_frequency"],
                mode="markers",
                # color depending on the exponential coeficient
                marker=dict(
                    color=exponential_coeficient,
                    size=10,
                    colorscale="Viridis",
                    colorbar=dict(
                        title="Exponential coeficient",
                    ),
                    showscale=True,
                    #  scale has to be positive
                    cmin=0.4,
                ),
                # name=p["roi_id"].values(),
                showlegend=False,
            )
        )

    return fig


@callback(
    Output("speed-tuning-plot", "children"),
    [
        Input("store", "data"),
        # Input("show-only-responsive", "checked"),
    ],
)
def speed_tuning_plot(store):
    if store == {}:
        return "No data to plot"

    data = load_data(store)
    data["n_neurons"]
    sfs = store["config"]["spatial_frequencies"]
    tfs = store["config"]["temporal_frequencies"]
    median_subtracted_response = data["median_subtracted_responses"]
    responsive_roi = data["responsive_neurons"]

    velocity = np.zeros((6, 6))
    for i, sf in enumerate(sfs):
        for j, tf in enumerate(tfs):
            velocity[i, j] = tf / sf

    flat_velocity = velocity.flatten()
    flat_velocity = np.round(flat_velocity, 4)

    fig = go.Figure()

    for roi in data["idx_neurons"]:
        key = (roi, "pooled")
        msr = median_subtracted_response[key]
        flat_msr = msr.flatten()
        unique_velocity = np.unique(flat_velocity)

        max_response_per_velocity = []
        for vel in unique_velocity:
            max_response_per_velocity.append(
                np.max(flat_msr[flat_velocity == vel])
            )

        fig.add_trace(
            go.Scatter(
                x=unique_velocity,
                y=max_response_per_velocity,
                mode="lines",
                marker=dict(
                    color="red" if roi in responsive_roi else "lightblue",
                    size=1,
                ),
                name=f"ROI {roi + 1}",
            )
        )

    fig.update_xaxes(type="log", title_text="Speed deg/s")
    fig.update_yaxes(title_text="Response ΔF/F")

    fig.update_layout(
        plot_bgcolor="white",
    )

    return dcc.Graph(figure=fig)


@callback(
    Output("murakami-plot2", "children"),
    [
        Input("store", "data"),
        Input("include-bad-fit", "checked"),
    ],
)
def murakami_plot_2(store: dict, include_bad_fit: bool) -> dcc.Graph:
    if store == {}:
        return "No data to plot"

    data = load_data(store)

    # prepare data
    n_neurons = data["n_neurons"]
    matrix_dimension = 100
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]
    median_subtracted_responses = data["median_subtracted_responses"]
    fit_outputs = data["fit_outputs"]

    neurons = data["idx_neurons"]

    exponential_coef = [
        fit_outputs[(roi_id, "pooled")][-1] for roi_id in data["idx_neurons"]
    ]

    fitted_gaussian_matrix = get_gaussian_matrix_to_be_plotted_for_all_rois(
        neurons,
        fit_outputs,
        spatial_frequencies,
        temporal_frequencies,
        matrix_dimension,
    )
    fitted_gaussian_matrix_small = (
        get_gaussian_matrix_to_be_plotted_for_all_rois(
            neurons,
            fit_outputs,
            spatial_frequencies,
            temporal_frequencies,
            6,
        )
    )
    #  get fit correlations for all rois
    fit_correlations = []
    for roi_id in data["idx_neurons"]:
        try:
            corr = fit_correlation(
                fitted_gaussian_matrix_small[(roi_id, "pooled")],
                median_subtracted_responses[(roi_id, "pooled")],
            )
            fit_correlations.append(corr)
        except Exception:
            # array must not contain infs or NaNs
            fit_correlations.append(0)

    #  only rois where fit correlationn > 0.75 and fit_values > 0.4
    total_roi = [
        roi_id
        for roi_id in range(n_neurons)
        if (exponential_coef[roi_id] > 0.2)
        & (include_bad_fit or (fit_correlations[roi_id] > 0.60))
    ]
    # for roi in total_roi:
    #     print(roi, fit_correlations[roi], exponential_coef[roi])

    if total_roi == []:
        return "No data to plot"

    neurons_idxs = data["idx_neurons"]

    fig = go.Figure()
    fig = add_data_in_figure(
        all_roi=neurons_idxs,
        fig=fig,
        matrix_dimension=matrix_dimension,
        fitted_gaussian_matrix=fitted_gaussian_matrix,
        spatial_frequencies=spatial_frequencies,
        temporal_frequencies=temporal_frequencies,
        marker_style="gaussian_fit",
        exponential_coeficient=exponential_coef,
    )
    fig = prettify_murakami_plot(
        fig, spatial_frequencies, temporal_frequencies
    )

    return dcc.Graph(figure=fig)
