from collections import namedtuple
import itertools
from pathlib import Path
import pickle
import dash
import dash_mantine_components as dmc
from dash import html, dcc, callback, Input, Output
import numpy as np
import pandas as pd
import plotly.express as px

from rsp_vision.objects.photon_data import PhotonData

dash.register_page(__name__, path="/sf_tf_facet_plot")

layout = html.Div(
    [
        dmc.Title(
            "SF-TF facet plot and gaussians",
            className="page-title",
        ),
        dmc.Grid(
            children=[
                dmc.Col(
                    [
                        dmc.Text(
                            id="selected_data_str_sf_tf",
                        ),
                        html.Br(),
                        # dmc.Switch(
                        #     id="show-only-responsive",
                        #     label="Show only responsive ROIs",
                        #     checked=True,
                        # ),
                        html.Br(),
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
                            label="SF-TF facet plot and gaussians",
                            href="/sf_tf_facet_plot",
                            className="navlink",
                            disabled=True,
                        ),
                        dmc.NavLink(
                            label="Polar plots",
                            href="/polar_plots",
                            className="navlink",
                        ),
                    ],
                    span=2,
                ),
                dmc.Col(
                    [
                        html.Div(
                            id="sf-tf-plot",
                            className="sf-tf-plot",
                        ),
                    ],
                    span="auto",
                ),
                dmc.Col(
                    [
                        html.Div(
                            id="gaussian-plot",
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


def load_data(store):
    path = (
        Path(store["path"])
        / store["subject_folder_path"]
        / store["session_folder_path"]
    )
    
    with open(path / "roi_0_signal_dataframe.pickle", "rb") as f:
        signal = pickle.load(f)

    return signal

@callback(
    Output("sf-tf-plot", "children"),
    Input("store", "data"),
    # [
    #     # Input("roi-choice-dropdown", "value"),
    #     # Input("direction-store", "data"),
    #     # Input("toggle-traces", "value"),
    # ],
)
def sf_tf_grid(store
    # roi_id: int, direction_input: dict, toggle_value: str
) -> dcc.Graph:
    if store == {}:
        return "No data to plot"

    signal = load_data(store)
    spatial_frequencies = store["config"]["spatial_frequencies"]
    temporal_frequencies = store["config"]["temporal_frequencies"]

    # direction = direction_input["value"]
    direction = 90
    roi_id = 0
    toggle_value = "ALL"
    counts = get_df_sf_tf_combo_plot(signal)

    #  from data I need: n_days, sf_tf_combinations
    sf_tf_combinations = itertools.product(
        spatial_frequencies, temporal_frequencies
    )
    total_n_days = signal.day.max()

    Data = namedtuple("data", ["sf_tf_combinations", "total_n_days"])
    data = Data(sf_tf_combinations, total_n_days)

    if direction == "all":
        dataframe = get_dataframe_for_facet_plot_pooled_directions(
            signal, roi_id
        )
    else:
        assert isinstance(direction, int)
        dataframe = get_dataframe_for_facet_plot(
            signal, data, counts, roi_id, direction
        )

    fig = px.line(
        dataframe,
        x="stimulus_frames",
        y="signal",
        facet_col="tf",
        facet_row="sf",
        facet_col_spacing=0.005,
        facet_row_spacing=0.005,
        # width=2000,
        height=1200,
        color="signal_kind",
        category_orders={
            "sf": spatial_frequencies[::-1],
            "tf": temporal_frequencies,
        },
    )

    fig.update_layout(
        title=f"SF TF traces for roi {roi_id + 1}",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
    )
    for trace in fig.data:
        if "mean" in trace.name:
            trace.line.color = "black"
            trace.line.width = 3
            trace.line.dash = "solid"
        elif "median" in trace.name:
            trace.line.color = "red"
            trace.line.width = 3
            trace.line.dash = "solid"
        else:
            if "ALL" not in toggle_value:
                trace.visible = False
            else:
                trace.line.width = 0.5

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
            annotation_font_size=15,
            fillcolor=color,
            opacity=0.1,
            line_width=0,
        )

    # Fake legend
    fig.add_annotation(
        x=0.9,
        y=0.97,
        xref="paper",
        yref="paper",
        text="mean",
        showarrow=False,
        font=dict(size=15, color="black"),
    )
    fig.add_annotation(
        x=0.95,
        y=0.97,
        xref="paper",
        yref="paper",
        text="median",
        showarrow=False,
        font=dict(size=15, color="red"),
    )

    return html.Div(
        dcc.Graph(
            id="sf_tf_plot",
            figure=fig,
        )
    )


def get_df_sf_tf_combo_plot(
    signal: pd.DataFrame
) -> np.ndarray:
    signal["stimulus_frames"] = np.nan
    n_frames_per_stim = int( 75 * 3
        # data.n_frames_per_trigger * data.n_triggers_per_stimulus
    )
    counts = np.arange(0, n_frames_per_stim)
    start_frames_indexes = signal[signal["stimulus_onset"]].index

    for idx in start_frames_indexes:
        start = idx
        end = idx + n_frames_per_stim - 1
        signal.loc[start:end, "stimulus_frames"] = counts
        signal.loc[start:end, "sf"] = signal.loc[idx, "sf"]
        signal.loc[start:end, "tf"] = signal.loc[idx, "tf"]
        signal.loc[start:end, "direction"] = signal.loc[idx, "direction"]
        signal.loc[start:end, "roi_id"] = signal.loc[idx, "roi_id"]
        signal.loc[start:end, "session_id"] = signal.loc[idx, "session_id"]

    return counts
    

def get_dataframe_for_facet_plot_pooled_directions(
    signal: pd.DataFrame,
    roi_id: int,
) -> pd.DataFrame:
    this_roi_df = signal[
        (signal["roi_id"] == roi_id)
        & signal.sf.notnull()
        & signal.tf.notnull()
    ]

    this_roi_df["signal_kind"] = (
        this_roi_df["session_id"].astype(str)
        + "_"
        + this_roi_df["direction"].astype(str)
    )
    this_roi_df.drop(columns=["session_id", "direction"], inplace=True)
    this_roi_df = this_roi_df[
        ["stimulus_frames", "signal", "sf", "tf", "signal_kind"]
    ]

    mean_df = (
        this_roi_df.groupby(["sf", "tf", "stimulus_frames"])
        .agg({"signal": "mean"})
        .reset_index()
    )
    mean_df["signal_kind"] = "mean"
    combined_df = pd.concat([this_roi_df, mean_df], ignore_index=True)

    median_df = (
        this_roi_df.groupby(["sf", "tf", "stimulus_frames"])
        .agg({"signal": "median"})
        .reset_index()
    )
    median_df["signal_kind"] = "median"
    combined_df = pd.concat([combined_df, median_df], ignore_index=True)

    return combined_df


def get_dataframe_for_facet_plot(
    signal: pd.DataFrame,
    data: PhotonData,
    counts: np.ndarray,
    roi_id: int,
    direction: int,
) -> pd.DataFrame:
    this_roi_df = signal[
        (signal["roi_id"] == roi_id)
        & signal.sf.notnull()
        & signal.tf.notnull()
    ]

    columns = [
        "stimulus_frames",
        "mean_signal",
        "median_signal",
        "sf",
        "tf",
        "dir",
    ]
    reps = [
        "signal_rep_1",
        "signal_rep_2",
        "signal_rep_3",
    ]
    if data.total_n_days > 1:
        reps += [
            "signal_rep_4",
            "signal_rep_5",
            "signal_rep_6",
        ]
    if data.total_n_days > 2:
        reps += [
            "signal_rep_7",
            "signal_rep_8",
            "signal_rep_9",
        ]
    columns += reps

    horizontal_df = pd.DataFrame(
        columns=columns,
    )

    for sf_tf in data.sf_tf_combinations:
        repetitions = this_roi_df[
            (this_roi_df.sf == sf_tf[0])
            & (this_roi_df.tf == sf_tf[1])
            & (this_roi_df.direction == direction)
        ]

        df = repetitions.pivot(index="stimulus_frames", columns="session_id")[
            "signal"
        ]
        cols = df.keys().values

        df.rename(
            columns=dict(zip(cols, reps)),
            inplace=True,
        )
        df["stimulus_frames"] = counts
        df["sf"] = repetitions.sf.iloc[0]
        df["tf"] = repetitions.tf.iloc[0]
        df["dir"] = repetitions.direction.iloc[0]
        df["mean_signal"] = df[reps].mean(axis=1)
        df["median_signal"] = df[reps].median(axis=1)

        horizontal_df = pd.concat([horizontal_df, df], ignore_index=True)

    vertical_df = pd.melt(
        horizontal_df,
        id_vars=[
            "stimulus_frames",
            "sf",
            "tf",
            "dir",
        ],
        value_vars=reps
        + [
            "mean_signal",
            "median_signal",
        ],
        var_name="signal_kind",
        value_name="signal",
    )

    return vertical_df
