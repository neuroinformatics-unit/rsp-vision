import pickle

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

app = Dash(__name__)

# LOAD DATA
# =============================================================================
with open("AK_1111739_hL_RSPd_monitor_front_analysis.pickle", "rb") as f:
    analysis = pickle.load(f)

data = analysis.data
signal = analysis.signal
responses = analysis.responses
p_values = analysis.p_values
magnitude = analysis.magintude_over_medians
responsive_rois = analysis.responsive_rois


rois = list(range(analysis.data.n_roi))


# ADAPT DATA FRAMES
# =============================================================================
def get_df_sf_tf_combo_plot():
    signal["stimulus_frames"] = np.nan
    n_frames_per_stim = int(
        data.n_frames_per_trigger * data.n_triggers_per_stimulus
    )
    counts = np.arange(0, n_frames_per_stim)
    start_frames_indexes = signal[signal["stimulus_onset"]].index

    for idx in start_frames_indexes:
        start = idx  # - analysis.padding[0]
        end = idx + n_frames_per_stim - 1  # + analysis.padding[1]
        signal.loc[start:end, "stimulus_frames"] = counts
        signal.loc[start:end, "sf"] = signal.loc[idx, "sf"]
        signal.loc[start:end, "tf"] = signal.loc[idx, "tf"]
        signal.loc[start:end, "direction"] = signal.loc[idx, "direction"]
        signal.loc[start:end, "roi_id"] = signal.loc[idx, "roi_id"]
        signal.loc[start:end, "session_id"] = signal.loc[idx, "session_id"]


get_df_sf_tf_combo_plot()
# print(signal.head())

# LAYOUT
# =============================================================================
app.layout = html.Div(
    [
        html.H1("RSP vision"),
        html.H3("Folder loaded is: AK_1111739_hL_RSPd_monitor_front"),
        html.H4("Choose a ROI"),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                options=[
                                    {"label": str(roi), "value": roi}
                                    for roi in rois
                                ],
                                value=0,
                                id="demo-dropdown",
                            ),
                        ),
                        dbc.Col(
                            [
                                html.H3("Responsive ROIs"),
                                html.Ul(
                                    [html.Li(x + 1) for x in responsive_rois]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        html.Div(
            id="example-graph",
        ),
        html.Div(
            id="sf_tf-graph",
        ),
    ]
)


# CALLBACKS
# =============================================================================
@app.callback(
    Output("example-graph", "children"), Input("demo-dropdown", "value")
)
def update_fig_all_sessions(roi):
    lineplot = px.line(
        signal[signal.roi_id == roi],
        x="frames_id",
        y="signal",
        color="session_id",
    )
    lineplot.update_traces(line=dict(width=0.3))

    scatterplot = px.scatter(
        signal[signal.stimulus_onset],
        x="frames_id",
        y="sf",
        color="tf",
    )

    return html.Div(
        dcc.Graph(
            id="my-graph",
            figure=go.Figure(data=lineplot.data + scatterplot.data),
        )
    )


@app.callback(
    Output("sf_tf-graph", "children"), Input("demo-dropdown", "value")
)
def sf_tf_grid(roi):
    no_nan = signal[
        (signal["roi_id"] == roi) & signal.sf.notnull() & signal.tf.notnull()
    ]

    fig = px.line(
        no_nan[no_nan["roi_id"] == roi],
        x="stimulus_frames",
        y="signal",
        facet_col="sf",
        facet_row="tf",
        width=1500,
        height=800,
        color="session_id",
    )
    fig.update_layout(title=f"SF TF traces for roi {roi}")
    fig.update_traces(line=dict(width=0.3))

    return html.Div(
        dcc.Graph(
            id="sf_tf_plot",
            figure=fig,
        )
    )


# RUN
# =============================================================================
if __name__ == "__main__":
    app.run_server(debug=True)
