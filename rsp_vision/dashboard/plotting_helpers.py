import math
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def get_df_sf_tf_combo_plot(signal, data):
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


def get_dataframe_for_facet_plot(signal, data, counts, roi_id, dir):
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

    for sf_tf in data.sf_tf_combinations:
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

    return vertical_df


def fit_correlation(gaussian, msr):
    fit_corr, _ = pearsonr(msr.flatten(), gaussian.flatten())
    return fit_corr


def find_peak_coordinates(oversampled_gaussians, sfs, tfs):
    # find the peak indices
    peak_indices = np.unravel_index(
        np.argmax(oversampled_gaussians), oversampled_gaussians.shape
    )

    # normalize the peak indices to the range [0,1]
    peak_norm = np.array(peak_indices) / np.array(oversampled_gaussians.shape)

    # map the normalized indices to octaves using
    # the min and max sf and tf values
    octaves = np.array(
        [
            from_frequency_to_octaves(peak_norm[0], sfs.min(), sfs.max()),
            from_frequency_to_octaves(peak_norm[1], tfs.min(), tfs.max()),
        ]
    )

    # if much smaller than sfs and tfs, then set the
    # peak to the smallest sf and tf
    if octaves[0] < from_frequency_to_octaves(
        np.min(sfs) / 5, sfs.min(), sfs.max()
    ):
        octaves[0] = from_frequency_to_octaves(
            np.min(sfs), sfs.min(), sfs.max()
        )
    if octaves[1] < from_frequency_to_octaves(
        np.min(tfs) / 5, tfs.min(), tfs.max()
    ):
        octaves[1] = from_frequency_to_octaves(
            np.min(tfs), tfs.min(), tfs.max()
        )

    # convert octaves to frequency values
    peak = from_octaves_to_frequency(octaves)

    return peak


def from_frequency_to_octaves(frequency, min_frequency, max_frequency):
    return np.log2(frequency / min_frequency) - np.log2(
        max_frequency / min_frequency
    )


def from_octaves_to_frequency(octaves):
    return 2**octaves


def generate_figure(
    directions: List[int],
    circle_x: List[float],
    circle_y: List[float],
    selected_direction: int,
) -> dict:
    return {
        "data": [
            {
                "type": "scatter",
                "x": circle_x,
                "y": circle_y,
                "mode": "markers+text",
                "text": [str(d) + "°" for d in directions],
                "textposition": "bottom center",
                "hoverinfo": "none",
                "marker": {"size": 10},
                "customdata": directions,
                "textfont": {
                    "color": [
                        "red" if d == selected_direction else "black"
                        for d in directions
                    ],
                    "size": [
                        15 if d == selected_direction else 10
                        for d in directions
                    ],
                    "weight": [
                        "bold" if d == selected_direction else "normal"
                        for d in directions
                    ],
                },
            }
        ],
        "layout": {
            "xaxis": {"range": [-1.2, 1.2], "visible": False},
            "yaxis": {"range": [-1.2, 1.2], "visible": False},
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "margin": {
                "l": 0,
                "r": 0,
                "t": 0,
                "b": 30,
                "pad": 0,
                "autoexpand": True,
            },
        },
    }


def get_circle_coordinates(
    directions,
):
    circle_x = [math.cos(math.radians(d)) for d in directions]
    circle_y = [math.sin(math.radians(d)) for d in directions]
    return circle_x, circle_y
