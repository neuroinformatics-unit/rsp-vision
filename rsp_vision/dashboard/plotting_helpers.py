import itertools
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rsp_vision.objects.photon_data import PhotonData


def get_df_sf_tf_combo_plot(
    signal: pd.DataFrame, data: PhotonData
) -> np.ndarray:
    signal["stimulus_frames"] = np.nan
    n_frames_per_stim = int(
        data.n_frames_per_trigger * data.n_triggers_per_stimulus
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
            & (this_roi_df.direction == direction)
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


def fit_correlation(
    gaussian: np.ndarray, median_subtracted_response: np.ndarray
) -> float:
    fit_corr, _ = pearsonr(
        median_subtracted_response.flatten(), gaussian.flatten()
    )
    return fit_corr


def find_peak_coordinates(
    oversampled_gaussian: np.ndarray,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    config: dict,
):
    # find the peak indices
    peak_indices = np.unravel_index(
        np.argmax(oversampled_gaussian), oversampled_gaussian.shape
    )

    # normalize the peak indices to the range [0,1]
    peak_norm = np.array(peak_indices) / np.array(oversampled_gaussian.shape)

    # Replace 0 frequency with a small value (min_frequency / 100)
    small_sf = (
        spatial_frequencies.min() / config["fitting"]["oversampling_factor"]
    )
    small_tf = (
        temporal_frequencies.min() / config["fitting"]["oversampling_factor"]
    )
    peak_norm[0] = peak_norm[0] if peak_norm[0] != 0 else small_sf
    peak_norm[1] = peak_norm[1] if peak_norm[1] != 0 else small_tf

    # map the normalized indices to octaves using
    # the min and max sf and tf values
    octaves = np.array(
        [
            from_frequency_to_octaves(
                peak_norm[0],
                spatial_frequencies.min(),
                spatial_frequencies.max(),
            ),
            from_frequency_to_octaves(
                peak_norm[1],
                temporal_frequencies.min(),
                temporal_frequencies.max(),
            ),
        ]
    )

    # convert octaves to frequency values
    peak = from_octaves_to_frequency(octaves)

    return peak


def from_frequency_to_octaves(
    frequency: float, min_frequency: float, max_frequency: float
) -> float:
    return np.log2(frequency / min_frequency) - np.log2(
        max_frequency / min_frequency
    )


def from_octaves_to_frequency(octaves: float) -> float:
    return 2**octaves


def get_direction_plot_for_controller(
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
    directions: np.ndarray,
) -> Tuple[List[float], List[float]]:
    circle_x = [math.cos(math.radians(d)) for d in directions]
    circle_y = [math.sin(math.radians(d)) for d in directions]
    return circle_x, circle_y


def get_corresponding_value(
    data: np.ndarray, roi_id: int, direction: int, sf_idx: int, tf_idx: int
) -> float:
    # if I use the oversampled gaussian, I get a different result
    # there is always a point in which the peak is very high
    # therefore it does not give us much information on the preference
    # of the neuron
    matrix = data[(roi_id, direction)]
    return matrix[tf_idx, sf_idx]


def get_peaks_dataframe(
    gaussian_or_original: str,
    roi_id: int,
    directions: np.ndarray,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    median_subtracted_responses: np.ndarray,
    downsampled_gaussians: np.ndarray,
) -> pd.DataFrame:
    if gaussian_or_original == "original":
        data = median_subtracted_responses
    elif gaussian_or_original == "gaussian":
        data = downsampled_gaussians

    p = pd.DataFrame(
        {
            "roi_id": roi_id,
            "direction": dire,
            "temporal_frequency": temporal_frequencies[tf_idx],
            "spatial_frequency": spatial_frequencies[sf_idx],
            "corresponding_value": get_corresponding_value(
                data, roi_id, dire, sf_idx, tf_idx
            ),
        }
        for dire in directions
        for tf_idx, sf_idx in itertools.product(
            range(len(temporal_frequencies)), range(len(spatial_frequencies))
        )
    )

    p_sorted = p.sort_values(by="direction")

    return p_sorted