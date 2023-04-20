import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


def get_update_fig_all_sessions_callback(
    app: Dash, signal: pd.DataFrame
) -> None:
    @app.callback(
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
