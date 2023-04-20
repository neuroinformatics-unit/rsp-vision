import numpy as np
import pandas as pd

from rsp_vision.analysis.gaussians_calculations import symmetric_2D_gaussian


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


# a bit slow
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


def get_preferred_sf_tf(responses, roi_id, dir):
    median_subtracted_response = (
        responses[(responses.roi_id == roi_id) & (responses.direction == dir)]
        .groupby(["sf", "tf"])[["subtracted"]]
        .median()
    )
    sf_0, tf_0 = median_subtracted_response["subtracted"].idxmax()
    peak_response = median_subtracted_response.loc[(sf_0, tf_0)]["subtracted"]
    return sf_0, tf_0, peak_response


def get_median_subtracted_response(responses, roi_id, dir, sfs_inverted, tfs):
    median_subtracted_response = (
        responses[(responses.roi_id == roi_id) & (responses.direction == dir)]
        .groupby(["sf", "tf"])[["subtracted"]]
        .median()
    )
    msr_for_plotting = np.zeros((len(sfs_inverted), len(tfs)))
    for i, sf in enumerate(sfs_inverted):
        for j, tf in enumerate(tfs):
            msr_for_plotting[i, j] = median_subtracted_response.loc[(sf, tf)][
                "subtracted"
            ]

    return msr_for_plotting


def fit_elliptical_gaussian(sfs_inverted, tfs, responses, roi_id, config, dir):
    sf_0, tf_0, peak_response = get_preferred_sf_tf(responses, roi_id, dir)

    # same tuning width for sf and tf
    sigma = config["fitting"]["tuning_width"]

    R = np.zeros((len(sfs_inverted), len(tfs)))
    for i, sf in enumerate(sfs_inverted):
        for j, tf in enumerate(tfs):
            R[i, j] = symmetric_2D_gaussian(
                peak_response, sf, tf, sf_0, tf_0, sigma
            )

    return R
