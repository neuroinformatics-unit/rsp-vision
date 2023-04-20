import pickle

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from rsp_vision.plots.utils import fit_elliptical_gaussian

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

    return counts


counts = get_df_sf_tf_combo_plot()
directions = list(data.uniques["direction"])

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
                            [
                                html.H3("Responsive ROIs"),
                                html.Ul(
                                    [html.Li(x + 1) for x in responsive_rois]
                                ),
                            ]
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                options=[
                                    {"label": str(roi + 1), "value": roi}
                                    for roi in rois
                                ],
                                value=8,
                                id="demo-dropdown",
                            ),
                        ),
                    ]
                ),
            ]
        ),
        html.Div(
            id="example-graph",
        ),
        html.Div(
            [
                html.H3("Directions"),
                dcc.RadioItems(
                    [{"label": str(dir), "value": dir} for dir in directions],
                    315.0,
                    id="directions-checkbox",
                ),
            ]
        ),
        html.Div(
            id="sf_tf-graph",
        ),
        html.Div(
            id="median-response-graph",
        ),
        html.Div(
            id="gaussian-graph",
        ),
    ]
)


# CALLBACKS
# =============================================================================
@app.callback(
    Output("example-graph", "children"), Input("demo-dropdown", "value")
)
def update_fig_all_sessions(roi_id):
    lineplot = px.line(
        signal[signal.roi_id == roi_id],
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
    fig = go.Figure(data=lineplot.data + scatterplot.data)
    fig.update_layout(
        title=f"Signal across sessions, roi: {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
    )

    return html.Div(
        dcc.Graph(
            id="my-graph",
            figure=fig,
        )
    )


@app.callback(
    Output("sf_tf-graph", "children"),
    [Input("demo-dropdown", "value"), Input("directions-checkbox", "value")],
)
def sf_tf_grid(roi_id, dir):
    this_roi_df = signal[
        (signal["roi_id"] == roi_id)
        & signal.sf.notnull()
        & signal.tf.notnull()
    ]

    horizontal_df = pd.DataFrame(
        columns=[
            "stimulus_frames",
            "signal_rep_1",
            "signal_rep_2",
            "signal_rep_3",
            "mean_signal",
            "median_signal",
            "sf",
            "tf",
            "dir",
        ]
    )

    for sf_tf in analysis.sf_tf_combinations:
        repetitions = this_roi_df[
            (this_roi_df.sf == sf_tf[0])
            & (this_roi_df.tf == sf_tf[1])
            & (this_roi_df.direction == dir)
        ]

        df = repetitions.pivot(index="stimulus_frames", columns="session_id")[
            "signal"
        ]
        cols = df.keys().values
        df.rename(
            columns={
                cols[0]: "signal_rep_1",
                cols[1]: "signal_rep_2",
                cols[2]: "signal_rep_3",
            },
            inplace=True,
        )
        df["stimulus_frames"] = counts
        df["sf"] = repetitions.sf.iloc[0]
        df["tf"] = repetitions.tf.iloc[0]
        df["dir"] = repetitions.direction.iloc[0]
        df["mean_signal"] = df[
            [
                "signal_rep_1",
                "signal_rep_2",
                "signal_rep_3",
            ]
        ].mean(axis=1)
        df["median_signal"] = df[
            [
                "signal_rep_1",
                "signal_rep_2",
                "signal_rep_3",
            ]
        ].median(axis=1)

        horizontal_df = pd.concat([horizontal_df, df], ignore_index=True)

    vertical_df = pd.melt(
        horizontal_df,
        id_vars=[
            "stimulus_frames",
            "sf",
            "tf",
            "dir",
        ],
        value_vars=[
            "signal_rep_1",
            "signal_rep_2",
            "signal_rep_3",
            "mean_signal",
            "median_signal",
        ],
        var_name="signal_kind",
        value_name="signal",
    )

    # print(vertical_df)

    fig = px.line(
        vertical_df,
        x="stimulus_frames",
        y="signal",
        facet_col="sf",
        facet_row="tf",
        width=1500,
        height=800,
        color="signal_kind",
    )

    fig.update_layout(
        title=f"SF TF traces for roi {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    fig.update_traces(line=dict(width=0.5))

    return html.Div(
        dcc.Graph(
            id="sf_tf_plot",
            figure=fig,
        )
    )


@app.callback(
    Output("median-response-graph", "children"),
    [Input("demo-dropdown", "value"), Input("directions-checkbox", "value")],
)
def responses_heatmap(roi_id, dir):
    median_subtracted_response = (
        responses[(responses.roi_id == roi_id) & (responses.direction == dir)]
        .groupby(["sf", "tf"])[["subtracted"]]
        .median()
    )
    sfs = np.sort(data.uniques["sf"])[::-1]
    tfs = np.sort(data.uniques["tf"])

    array = np.zeros((len(sfs), len(tfs)))
    for i, sf in enumerate(sfs):
        for j, tf in enumerate(tfs):
            array[i, j] = median_subtracted_response.loc[(sf, tf)][
                "subtracted"
            ]

    y_labels = list(map(str, sfs.tolist()))
    x_labels = list(map(str, tfs.tolist()))

    # array = median_subtracted_response["subtracted"].values.reshape(6, 6)

    fig = px.imshow(array, x=x_labels, y=y_labels)

    return html.Div(
        dcc.Graph(
            id="median_response_plot",
            figure=fig,
        )
    )


@app.callback(
    Output("gaussian-graph", "children"),
    [Input("demo-dropdown", "value"), Input("directions-checkbox", "value")],
)
def gaussian_plot(roi_id, dir):
    R = fit_elliptical_gaussian(
        data.uniques, responses, roi_id, data.config, dir
    )
    y_labels = list(map(str, np.sort(data.uniques["sf"]).tolist()))
    x_labels = list(map(str, np.sort(data.uniques["tf"]).tolist()))
    fig = px.imshow(R, x=x_labels, y=y_labels, aspect="equal")

    return html.Div(
        dcc.Graph(
            id="gaussian_plot",
            figure=fig,
        )
    )


# RUN
# =============================================================================
if __name__ == "__main__":
    app.run_server(debug=True)
