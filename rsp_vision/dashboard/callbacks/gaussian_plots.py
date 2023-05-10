from typing import Dict, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import Input, Output, callback, dcc, html

from rsp_vision.dashboard.callbacks.plotting_helpers import fit_correlation


def get_andermann_gaussian_plot_callback(
    median_subtracted_responses: np.ndarray,
    downsampled_gaussians: Dict[Tuple[int, Union[int, str]], np.ndarray],
    oversampled_gaussians: Dict[Tuple[int, Union[int, str]], np.ndarray],
    fit_outputs: np.ndarray,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
) -> None:
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
