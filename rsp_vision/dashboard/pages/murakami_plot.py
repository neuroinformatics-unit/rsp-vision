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
                ),
                dmc.Col(
                    html.Div(
                        id="speed-tuning-plot",
                        className="speed-tuning-plot",
                    ),
                    span="auto",
                ),
                dmc.Col(
                    html.Div(
                        id="frequency-tuning-plot",
                        className="frequency-tuning-plot",
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
        Input("include-bad-fit", "checked"),
    ],
)
def murakami_plot(
    store: dict,
    show_only_responsive: bool = True,
    include_bad_fit: bool = False,
) -> dcc.Graph:
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

    dataset = ExtractedData(
        store,
        is_responsive_only=show_only_responsive,
        include_bad_fit=include_bad_fit,
    )
    if dataset.choosen_roi == []:
        return "No data to plot"

    fig = go.Figure()
    fig = add_data_in_figure(
        all_roi=dataset.choosen_roi,
        fig=fig,
        matrix_dimension=dataset.matrix_dimension,
        responsive_rois=dataset.responsive_rois,
        fitted_gaussian_matrix=dataset.fitted_gaussian_matrix,
        spatial_frequencies=dataset.spatial_frequencies,
        temporal_frequencies=dataset.temporal_frequencies,
        exponential_coeficient=dataset.xi,
    )
    fig = prettify_murakami_plot(
        fig, dataset.spatial_frequencies, dataset.temporal_frequencies
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
        range=[-0.5, 7.5],
        title_text="Temporal frequency (Hz)",
        side="bottom",
        tickvals=[],
    )
    fig.update_yaxes(
        range=[-1, 7.5],
        title_text="Spatial frequency (cycles/deg)",
        tickvals=[],
    )

    for i in range(1, 7):
        fig.add_shape(
            type="line",
            x0=0.5,
            y0=i,
            x1=6.5,
            y1=i,
            line=dict(color="Grey", width=1, dash="dot"),
        )
        #  add annotations for horizontal lines
        fig.add_annotation(
            x=0,
            y=i,
            text=f"{spatial_frequencies[i-1]:.2f}",
            showarrow=False,
            yshift=0,
            xshift=-1,
            font=dict(color="Black"),
        )

        fig.add_shape(
            type="line",
            x0=i,
            y0=0.5,
            x1=i,
            y1=6.5,
            line=dict(color="Grey", width=1, dash="dot"),
        )
        #  add annotations for vertical lines
        fig.add_annotation(
            x=i,
            y=0,
            text=f"{temporal_frequencies[i-1]:.2f}",
            showarrow=False,
            yshift=-1,
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
    {
        "temporal_frequency": frequency_to_octaves_for_plotting(
            median_peaks["temporal_frequency"], temporal_frequencies
        ),
        "spatial_frequency": frequency_to_octaves_for_plotting(
            median_peaks["spatial_frequency"], spatial_frequencies
        ),
    }
    # for roi_id in all_roi:
    #     row = p[(p.roi_id == roi_id)].iloc[0]
    #     tf_oct = frequency_to_octaves_for_plotting(
    #             row["temporal_frequency"], temporal_frequencies
    #         )
    #     sf_oct = frequency_to_octaves_for_plotting(
    #             row["spatial_frequency"], spatial_frequencies
    #         )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=[tf_oct,
    #                median_peaks_oct["temporal_frequency"]],
    #             y=[sf_oct,
    #                median_peaks_oct["spatial_frequency"]],
    #             mode="lines",
    #             line=dict(color="Grey", width=1),
    #             showlegend=False,
    #         )
    #     )

    fig.add_trace(
        go.Scatter(
            x=[
                frequency_to_octaves_for_plotting(t, temporal_frequencies)
                for t in p["temporal_frequency"]
            ],
            y=[
                frequency_to_octaves_for_plotting(s, spatial_frequencies)
                for s in p["spatial_frequency"]
            ],
            mode="markers",
            text=np.asarray(p["roi_id"]) + 1,
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
                cmin=-1,
                cmax=1,
            ),
            showlegend=False,
        )
    )

    return fig


@callback(
    Output("speed-tuning-plot", "children"),
    [
        Input("store", "data"),
        Input("show-only-responsive", "checked"),
        Input("include-bad-fit", "checked"),
    ],
)
def speed_tuning_plot(store, show_only_responsive, include_bad_fit):
    if store == {}:
        return "No data to plot"

    dataset = ExtractedData(store, show_only_responsive, include_bad_fit)
    if dataset.choosen_roi == []:
        return "No data to plot"

    flat_velocity = make_velocity_array(
        dataset.spatial_frequencies, dataset.temporal_frequencies
    )

    fig = go.Figure()

    max_found = 0
    for roi in dataset.choosen_roi:
        unique_velocity, max_response_per_velocity = get_mean_velocity(
            dataset.msr[(roi, "pooled")], flat_velocity
        )

        fig.add_trace(
            go.Scatter(
                x=unique_velocity,
                y=max_response_per_velocity,
                mode="lines",
                marker=dict(
                    color="red"
                    if roi in dataset.responsive_rois
                    else "lightblue",
                    size=1,
                ),
                name=f"ROI {roi + 1}",
            )
        )
        if max(max_response_per_velocity) > max_found:
            max_found = max(max_response_per_velocity)

    fig.update_xaxes(type="log", title_text="Speed deg/s")
    fig.update_yaxes(title_text="Response ΔF/F")

    fig.update_layout(
        plot_bgcolor="white",
    )

    # fix fig size
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    #  draw vertical lines at given velocity
    for v in np.unique(flat_velocity):
        fig.add_shape(
            type="line",
            x0=v,
            y0=0,
            x1=v,
            y1=max_found,
            line=dict(color="Grey", width=1),
        )

    fig.update_xaxes(
        tickvals=np.unique(flat_velocity),
        ticktext=[f"{v}" for v in np.unique(flat_velocity)],
    )

    return dcc.Graph(figure=fig)


@callback(
    Output("frequency-tuning-plot", "children"),
    [
        Input("store", "data"),
        Input("show-only-responsive", "checked"),
        Input("include-bad-fit", "checked"),
    ],
)
def frequency_tuning_plot(store, show_only_responsive, include_bad_fit):
    if store == {}:
        return "No data to plot"

    dataset = ExtractedData(store, show_only_responsive, include_bad_fit)
    if dataset.choosen_roi == []:
        return "No data to plot"

    # make figure with two subplots
    #  first: spatial frequency mean values vs ΔF/f
    #  second: temporal frequency mean values vs ΔF/f

    fig = go.Figure()

    relative_max = 0
    for roi in dataset.choosen_roi:
        #  spatial frequency
        fig.add_trace(
            go.Scatter(
                x=dataset.spatial_frequencies,
                y=np.mean(dataset.msr[(roi, "pooled")], axis=0),
                mode="lines",
                marker=dict(
                    color="red"
                    if roi in dataset.responsive_rois
                    else "lightblue",
                    size=1,
                ),
                name=f"ROI {roi + 1}",
            )
        )

        #  temporal frequency
        fig.add_trace(
            go.Scatter(
                x=dataset.temporal_frequencies,
                y=np.mean(dataset.msr[(roi, "pooled")], axis=1),
                mode="lines",
                marker=dict(
                    color="red"
                    if roi in dataset.responsive_rois
                    else "lightblue",
                    size=1,
                ),
                name=f"ROI {roi + 1}",
            )
        )

        if (
            np.max(np.mean(dataset.msr[(roi, "pooled")], axis=0))
            > relative_max
        ):
            relative_max = np.max(dataset.msr[(roi, "pooled")])

    fig.update_xaxes(
        type="log",
        title_text="SF (cycles/deg)            |            TF (Hz)",
    )

    fig.update_yaxes(title_text="Response ΔF/F")
    fig.update_xaxes(
        tickvals=dataset.spatial_frequencies + dataset.temporal_frequencies,
        ticktext=[f"{sf:.2f}" for sf in dataset.spatial_frequencies],
    )
    fig.update_layout(
        plot_bgcolor="white",
    )

    # fix fig size
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    #  add vertical lines at given frequencies
    for sf in dataset.spatial_frequencies:
        fig.add_shape(
            type="line",
            x0=sf,
            y0=0,
            x1=sf,
            y1=relative_max,
            line=dict(color="Grey", width=1),
        )

    for tf in dataset.temporal_frequencies:
        fig.add_shape(
            type="line",
            x0=tf,
            y0=0,
            x1=tf,
            y1=relative_max,
            line=dict(color="Grey", width=1),
        )

    return dcc.Graph(figure=fig)


class ExtractedData:
    def __init__(self, store, is_responsive_only=True, include_bad_fit=False):
        self.data = load_data(store)
        self.responsive_rois = self.data["responsive_neurons"]
        self.n_neurons = self.data["n_neurons"]
        self.neurons_idx = self.data["idx_neurons"]
        self.matrix_dimension = 100
        self.spatial_frequencies = store["config"]["spatial_frequencies"]
        self.temporal_frequencies = store["config"]["temporal_frequencies"]

        self.msr = self.data["median_subtracted_responses"]

        self.fit_outputs = self.data["fit_outputs"]
        self.fitted_gaussian_matrix = (
            get_gaussian_matrix_to_be_plotted_for_all_rois(
                self.neurons_idx,
                self.fit_outputs,
                self.spatial_frequencies,
                self.temporal_frequencies,
                self.matrix_dimension,
            )
        )

        self.xi = [
            self.fit_outputs[(roi_id, "pooled")][-1]
            for roi_id in self.neurons_idx
        ]

        self.fitted_gaussian_matrix_small = (
            get_gaussian_matrix_to_be_plotted_for_all_rois(
                neuron_idxs=self.neurons_idx,
                fit_outputs=self.fit_outputs,
                spatial_frequencies=self.spatial_frequencies,
                temporal_frequencies=self.temporal_frequencies,
                matrix_dimension=6,
                kind="6x6 matrix",
            )
        )

        self.choosen_roi = (
            self.neurons_idx
            if not is_responsive_only
            else self.responsive_rois
        )

        self.fit_correlations = {}
        for roi_id in self.choosen_roi:
            try:
                corr = fit_correlation(
                    self.fitted_gaussian_matrix_small[(roi_id, "pooled")],
                    self.msr[(roi_id, "pooled")],
                )
                self.fit_correlations[roi_id] = corr
            except Exception:
                # array must not contain infs or NaNs
                self.fit_correlations[roi_id] = 0

        self.choosen_roi = [
            roi_id
            for roi_id in self.choosen_roi
            if (self.fit_correlations[roi_id] >= 0.80) or (include_bad_fit)
        ]


def make_velocity_array(sfs, tfs):
    velocity = np.zeros((6, 6))
    for i, sf in enumerate(sfs):
        for j, tf in enumerate(tfs):
            velocity[i, j] = tf / sf

    flat_velocity = velocity.flatten()
    flat_velocity = np.round(flat_velocity, 4)

    return flat_velocity


def get_mean_velocity(msr, flat_velocity):
    flat_msr = msr.flatten()
    unique_velocity = np.unique(flat_velocity)

    max_response_per_velocity = []
    for vel in unique_velocity:
        max_response_per_velocity.append(
            np.mean(flat_msr[flat_velocity == vel])
        )

    return unique_velocity, max_response_per_velocity


def frequency_to_octaves_for_plotting(val, frequencies):
    freq = np.asarray(frequencies)
    freq = np.concatenate((freq, [freq[-1] * 2]))
    closest_index = np.argmin(np.abs(freq - val))
    if val >= freq[closest_index]:
        low_index = closest_index
        high_index = low_index + 1
    else:
        high_index = closest_index
        low_index = high_index - 1

    space_decimal = np.linspace(freq[low_index], freq[high_index], 10)
    log_distance = np.argmin(np.abs(space_decimal - val))
    idx = 1 + low_index + log_distance / 10
    return idx
