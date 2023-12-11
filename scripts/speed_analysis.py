import itertools

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from rsp_vision.analysis.gaussians_calculations import (
    create_gaussian_matrix,
    make_space,
)

spatial_frequencies = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
temporal_frequencies = [0.5, 1, 2, 4, 8, 16]
matrix_dimension = 6


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

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Heatmap(
            z=gaussian_matrix,
            x=uniform_oversampled_tfs,
            y=uniform_oversampled_sfs,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=1,
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
        ticktext=np.round(longer_array_sfs, 2),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        tickvals=uniform_oversampled_tfs,
        ticktext=np.round(longer_array_tfs, 2),
        row=1,
        col=1,
    )

    # add x and y labels
    fig.update_xaxes(title_text="Temporal Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Spatial Frequency", row=1, col=1)

    #  maintain heatmap aspect ratio row 1 col 1
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        paper_bgcolor="white",
    )

    velocity = np.zeros((matrix_dimension, matrix_dimension))
    for i, sf in enumerate(space_sfs):
        for j, tf in enumerate(space_tfs):
            velocity[i, j] = sf / tf

    #  write numbers on the heatmap in row 1 col 2
    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                x=j,
                y=i,
                text=str(np.round(velocity[i, j], 4)),
                showarrow=False,
                font=dict(color="black", size=10),
            )
            for i, j in itertools.product(
                range(matrix_dimension), range(matrix_dimension)
            )
        ]
    )

    flat_velocity = velocity.flatten()
    flat_velocity = np.round(flat_velocity, 4)
    unique_velocity = np.unique(flat_velocity)

    flat_gaussian_matrix = gaussian_matrix.flatten()

    #  scatter plot of velocity vs response
    fig.add_trace(
        go.Scatter(
            x=flat_velocity,
            y=flat_gaussian_matrix,
            mode="markers",
            marker=dict(
                color="lightblue",
                size=5,
            ),
            name="Response",
        ),
        row=1,
        col=2,
    )

    #  x axis is in log scale
    fig.update_xaxes(type="log", title_text="Speed deg/s", row=1, col=2)
    fig.update_yaxes(title_text="Response ΔF/F", row=1, col=2)

    fig.update_layout(
        plot_bgcolor="white",
    )

    max_response_per_velocity = []
    for vel in unique_velocity:
        max_response_per_velocity.append(
            np.max(flat_gaussian_matrix[flat_velocity == vel])
        )

    fig.add_trace(
        go.Scatter(
            x=unique_velocity,
            y=max_response_per_velocity,
            mode="lines",
            marker=dict(
                color="red",
                size=5,
            ),
            name="Max Response",
        ),
        row=1,
        col=2,
    )

    min_response_per_velocity = []
    for vel in unique_velocity:
        min_response_per_velocity.append(
            np.min(flat_gaussian_matrix[flat_velocity == vel])
        )

    fig.add_trace(
        go.Scatter(
            x=unique_velocity,
            y=min_response_per_velocity,
            mode="lines",
            marker=dict(
                color="orange",
                size=5,
            ),
            name="Min Response",
        ),
        row=1,
        col=2,
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
