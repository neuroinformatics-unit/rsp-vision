import pickle

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

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
    signal["stimulus_frames"] = 0
    n_frames_per_stim = int(
        data.n_frames_per_trigger * data.n_triggers_per_stimulus
    )
    counts = np.arange(0, n_frames_per_stim + 1)
    start_frames_indexes = signal[signal["stimulus_onset"]].index

    for idx in start_frames_indexes:
        start = idx  # - analysis.padding[0]
        end = idx + n_frames_per_stim  # + analysis.padding[1]
        signal.loc[start:end, "stimulus_frames"] = counts


get_df_sf_tf_combo_plot()
print(signal.head())

# LAYOUT
# =============================================================================
app.layout = html.Div(
    [
        html.H1("RSP vision"),
        html.H3("Folder loaded is: AK_1111739_hL_RSPd_monitor_front"),
        html.H4("Choose a ROI"),
        dcc.Dropdown(
            options=[{"label": str(roi), "value": roi} for roi in rois],
            value=0,
            id="demo-dropdown",
        ),
        html.Div(
            id="example-graph",
        ),
        html.Center(
            [
                dash_table.DataTable(
                    p_values.to_dict("records"),
                    [{"name": i, "id": i} for i in p_values.columns],
                ),
                html.H3("Responsive ROIs"),
                html.Ul([html.Li(x + 1) for x in responsive_rois]),
            ]
        ),
    ]
)


# CALLBACKS
# =============================================================================
@app.callback(
    Output("example-graph", "children"), Input("demo-dropdown", "value")
)
def update_fig(roi):
    lineplot = px.line(
        signal[signal.roi_id == roi],
        x="frames_id",
        y="signal",
        color="session_id",
    )
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


# RUN
# =============================================================================
if __name__ == "__main__":
    app.run_server(debug=True)
