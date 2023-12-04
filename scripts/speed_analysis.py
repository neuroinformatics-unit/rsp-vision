import itertools

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from scipy.ndimage import rotate

from rsp_vision.analysis.gaussians_calculations import (
    create_gaussian_matrix,
    make_space,
)

spatial_frequencies = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
temporal_frequencies = [0.5, 1, 2, 4, 8, 16]
matrix_dimension = 10


uniform_oversampled_sfs = np.linspace(
    0, matrix_dimension - 1, matrix_dimension
)
uniform_oversampled_tfs = np.linspace(
    0, matrix_dimension - 1, matrix_dimension
)

# Params to change
peak_response = 2
sf_0 = 0.04
tf_0 = 4
sigma_sf = 1
sigma_tf = 1
𝜻_power_law_exp = 1


app = Dash()
app.layout = html.Div(
    [
        dcc.Graph(figure=go.Figure(), id="graph"),
        #  now add sliders
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Peak Response"),
                        dcc.Slider(
                            id="peak_response",
                            min=0,
                            max=100,
                            step=10,
                            value=peak_response,
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Label("SF_0"),
                        dcc.Slider(
                            id="sf_0",
                            min=min(spatial_frequencies),
                            max=max(spatial_frequencies),
                            step=0.02,
                            value=sf_0,
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Label("TF_0"),
                        dcc.Slider(
                            id="tf_0",
                            min=min(temporal_frequencies),
                            max=max(temporal_frequencies),
                            step=0.5,
                            value=tf_0,
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Label("Sigma SF"),
                        dcc.Slider(
                            id="sigma_sf",
                            min=0,
                            max=3,
                            step=0.2,
                            value=sigma_sf,
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Label("Sigma TF"),
                        dcc.Slider(
                            id="sigma_tf",
                            min=0,
                            max=3,
                            step=0.2,
                            value=sigma_tf,
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Label("𝜻 Power Law Exp"),
                        dcc.Slider(
                            id="𝜻_power_law_exp",
                            min=0,
                            max=2,
                            step=0.2,
                            value=𝜻_power_law_exp,
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Label("Speed Tuning"),
                        dcc.Graph(figure=go.Figure(), id="speed_tuning"),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
            ]
        ),
    ]
)


@app.callback(
    Output("graph", "figure"),
    Input("peak_response", "value"),
    Input("sf_0", "value"),
    Input("tf_0", "value"),
    Input("sigma_sf", "value"),
    Input("sigma_tf", "value"),
    Input("𝜻_power_law_exp", "value"),
)
def make_figure(
    peak_response, sf_0, tf_0, sigma_sf, sigma_tf, 𝜻_power_law_exp
):
    params = (peak_response, sf_0, tf_0, sigma_sf, sigma_tf, 𝜻_power_law_exp)

    space_sfs = make_space(
        spatial_frequencies,
        matrix_dimension,
        is_log=True,
    )
    space_tfs = make_space(
        temporal_frequencies,
        matrix_dimension,
        is_log=True,
    )

    gaussian_matrix = create_gaussian_matrix(params, space_sfs, space_tfs)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=gaussian_matrix,
            x=uniform_oversampled_tfs,
            y=uniform_oversampled_sfs,
            colorscale="Viridis",
            showscale=False,
            colorbar=dict(
                x=0.5,
                y=-0.1,
                xanchor="center",
                yanchor="top",
                len=1,
                orientation="h",
            ),
        ),
    )

    longer_array_sfs = make_space(
        spatial_frequencies,
        matrix_dimension,
        is_log=True,
    )
    longer_array_tfs = make_space(
        temporal_frequencies,
        matrix_dimension,
        is_log=True,
    )

    fig.update_yaxes(
        tickvals=uniform_oversampled_sfs,
        ticktext=np.round(longer_array_sfs[::10], 2),
    )

    fig.update_xaxes(
        tickvals=uniform_oversampled_tfs,
        ticktext=np.round(longer_array_tfs[::10], 2),
    )

    # add x and y labels
    fig.update_xaxes(title_text="Temporal Frequency")
    fig.update_yaxes(title_text="Spatial Frequency")

    #  remove legend
    fig.update_layout(showlegend=False)

    #  maintain heatmap aspect ratio
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )
    return fig


@app.callback(
    Output("speed_tuning", "figure"),
    Input("peak_response", "value"),
    Input("sf_0", "value"),
    Input("tf_0", "value"),
    Input("sigma_sf", "value"),
    Input("sigma_tf", "value"),
    Input("𝜻_power_law_exp", "value"),
)
def speed_tuning_plot(
    peak_response, sf_0, tf_0, sigma_sf, sigma_tf, 𝜻_power_law_exp
):
    params = (peak_response, sf_0, tf_0, sigma_sf, sigma_tf, 𝜻_power_law_exp)

    space_sfs = make_space(
        spatial_frequencies,
        matrix_dimension,
        is_log=True,
    )
    space_tfs = make_space(
        temporal_frequencies,
        matrix_dimension,
        is_log=True,
    )

    gaussian_matrix = create_gaussian_matrix(params, space_sfs, space_tfs)

    #  rotate matrix of 45 degrees

    rotated_gaussian_matrix = rotate(gaussian_matrix, -45, reshape=False)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=rotated_gaussian_matrix,
            x=uniform_oversampled_tfs,
            y=uniform_oversampled_sfs,
            colorscale="Viridis",
            showscale=False,
            colorbar=dict(
                x=0.5,
                y=-0.1,
                xanchor="center",
                yanchor="top",
                len=1,
                orientation="h",
            ),
        ),
    )
    #  axis off
    # fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    sum_of_rows = np.sum(rotated_gaussian_matrix, axis=0)
    #  divide by the number of not null values
    sum_of_rows = sum_of_rows / np.count_nonzero(sum_of_rows)

    fig.add_trace(
        go.Scatter(
            x=uniform_oversampled_tfs,
            y=sum_of_rows,
            mode="lines",
            name="Speed Tuning",
        ),
    )

    speeds = np.asarray(
        [sf / tf for sf, tf in itertools.product(space_sfs, space_tfs)]
    ).reshape(matrix_dimension, matrix_dimension)
    # print(np.round(speeds, 2))
    rotated_speeds = rotate(speeds, -45, reshape=False)
    print(np.round(rotated_speeds, 2))
    speed_diagonal = speeds[3]

    fig.update_xaxes(
        tickvals=uniform_oversampled_tfs,
        ticktext=np.round(speed_diagonal, 3),
    )

    #  x axis label
    fig.update_xaxes(title_text="Speed deg/s")

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
