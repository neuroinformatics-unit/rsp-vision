import logging

import numpy as np
import pandas as pd
import scipy.stats as ss

from ..objects.enums import PhotonType
from ..objects.photon_data import PhotonData
from .utils import get_fps


class SF_TF:
    def __init__(self, data: PhotonData, photon_type: PhotonType):
        self.data = data
        self.photon_type = photon_type
        self.fps = get_fps(photon_type, data.config)

        self.padding_start = int(data.config["padding"][0])
        self.padding_end = int(data.config["padding"][1])

        self.stimulus_idxs = self.data.signal[
            self.data.signal["stimulus_onset"]
        ].index

        self.adapted_signal = self.data.signal
        self.adapted_signal["mean_response"] = np.nan
        self.adapted_signal["mean_baseline"] = np.nan

    def get_response_and_baseline_windows(self):
        baseline_start = 0
        if self.data.n_triggers_per_stimulus == 3:
            response_start = 2
            if self.data.config["baseline"] == "static":
                baseline_start = 1
        if self.data.n_triggers_per_stimulus == 2:
            response_start = 1

        # Identify the rows in the window of each stimulus onset
        window_start_response = (
            self.stimulus_idxs
            + (self.data.n_frames_per_trigger * response_start)
            + (self.fps * 0.5)  # ignore first 0.5s
            - 1
        )
        window_end_response = (
            self.stimulus_idxs
            + (self.data.n_frames_per_trigger * (response_start + 1))
            - 1
        )
        window_mask_response = np.vstack(
            [
                np.arange(start, end)
                for start, end in zip(
                    window_start_response, window_end_response
                )
            ]
        )

        window_start_baseline = (
            self.stimulus_idxs
            + (self.data.n_frames_per_trigger * baseline_start)
            + (self.fps * 1.5)  # ignore first 1.5s
            - 1
        )
        window_end_baseline = (
            self.stimulus_idxs
            + (self.data.n_frames_per_trigger * (baseline_start + 1))
            - 1
        )
        window_mask_baseline = np.vstack(
            [
                np.arange(start, end)
                for start, end in zip(
                    window_start_baseline, window_end_baseline
                )
            ]
        )

        return window_mask_response, window_mask_baseline

    def calculate_mean_response_and_baseline(
        self,
    ):
        logging.info("Start to edit the signal dataframe...")

        (
            self.window_mask_response,
            self.window_mask_baseline,
        ) = self.get_response_and_baseline_windows()

        mean_response_signal = [
            np.mean(
                self.adapted_signal.iloc[
                    self.window_mask_response[i]
                ].signal.values
            )
            for i in range(len(self.window_mask_response))
        ]
        mean_baseline_signal = [
            np.mean(
                self.adapted_signal.iloc[
                    self.window_mask_baseline[i]
                ].signal.values
            )
            for i in range(len(self.window_mask_baseline))
        ]

        self.adapted_signal.loc[
            self.stimulus_idxs, "mean_response"
        ] = mean_response_signal
        self.adapted_signal.loc[
            self.stimulus_idxs, "mean_baseline"
        ] = mean_baseline_signal

        self.adapted_signal["subtracted"] = (
            self.adapted_signal["mean_response"]
            - self.adapted_signal["mean_baseline"]
        )

        self.responses = self.adapted_signal[
            self.adapted_signal["stimulus_onset"]
        ][
            [
                "frames_id",
                "roi_id",
                "session_id",
                "sf",
                "tf",
                "direction",
                "mean_response",
                "mean_baseline",
                "subtracted",
            ]
        ]
        self.responses = self.responses.reset_index()

    def get_fit_parameters(self):
        # calls _fit_two_dimensional_elliptical_gaussian
        raise NotImplementedError("This method is not implemented yet")

    def responsiveness(self):
        self.calculate_mean_response_and_baseline()
        logging.info(f"Adapted signal dataframe:{self.responses.head()}")

        self.p_values = pd.DataFrame(
            columns=[
                "Kruskal-Wallis test",
                "Sign test",
                "Wilcoxon signed rank test",
            ]
        )

        self.p_values["Kruskal-Wallis test"] = self.nonparam_anova_over_rois()
        (
            self.p_values["Sign test"],
            self.p_values["Wilcoxon signed rank test"],
        ) = self.are_responses_significant()
        logging.info(f"P-values for each roi:\n{self.p_values}")

        self.magintude_over_medians = self.response_magnitude()

    def response_magnitude(self):
        # get specific windows for each sf/tf combo

        magintude_over_medians = pd.DataFrame(
            columns=[
                "roi",
                "sf",
                "tf",
                "response_mean",
                "baseline_mean",
                "baseline_std",
                "magnitude",
            ]
        )

        for roi in range(self.data.n_roi):
            for i, sf_tf in enumerate(self.sf_tf_combinations):
                sf_tf_idx = self.responses[
                    (self.responses.sf == sf_tf[0])
                    & (self.responses.tf == sf_tf[1])
                    & (self.responses.roi_id == roi)
                ].index
                r_windows = self.window_mask_response[sf_tf_idx]
                b_windows = self.window_mask_baseline[sf_tf_idx]

                responses_dir_and_reps = np.zeros(
                    (r_windows.shape[0], r_windows.shape[1])
                )
                baseline_dir_and_reps = np.zeros(
                    (b_windows.shape[0], b_windows.shape[1])
                )

                for i, w in enumerate(r_windows):
                    responses_dir_and_reps[i, :] = self.adapted_signal.iloc[
                        w
                    ].signal.values

                for i, w in enumerate(b_windows):
                    baseline_dir_and_reps[i, :] = self.adapted_signal.iloc[
                        w
                    ].signal.values

                median_response = np.median(responses_dir_and_reps, axis=0)
                median_baseline = np.median(baseline_dir_and_reps, axis=0)

                m_r = np.mean(median_response)
                m_b = np.mean(median_baseline)
                std_b = np.std(median_baseline, ddof=1)
                magnitude = (m_r - m_b) / std_b

                df = pd.DataFrame(
                    {
                        "roi": roi,
                        "sf": sf_tf[0],
                        "tf": sf_tf[1],
                        "response_mean": m_r,
                        "baseline_mean": m_b,
                        "baseline_std": std_b,
                        "magnitude": magnitude,
                    },
                    index=[0],
                )

                magintude_over_medians = pd.concat(
                    [magintude_over_medians, df], ignore_index=True
                )

        return magintude_over_medians

    def nonparam_anova_over_rois(self) -> dict:
        # Use Kruskal-Wallis H Test because it is nonparametric.
        # Compare if more than two independent
        # samples have a different distribution

        p_values = {}
        for roi in range(self.data.n_roi):
            melted = pd.melt(
                self.responses[self.responses.roi_id == roi],
                id_vars=["sf", "tf"],
                value_vars=["subtracted"],
            )

            self._sf = np.sort(self.data.uniques["sf"])
            self._tf = np.sort(self.data.uniques["tf"])
            self.sf_tf_combinations = np.array(
                np.meshgrid(self._sf, self._tf)
            ).T.reshape(-1, 2)
            self._dir = self.data.uniques["direction"]

            samples = np.zeros(
                (
                    len(self._dir) * self.data.n_triggers_per_stimulus,
                    len(self.sf_tf_combinations),
                )
            )

            for i, sf_tf in enumerate(self.sf_tf_combinations):
                samples[:, i] = melted[
                    (melted.sf == sf_tf[0]) & (melted.tf == sf_tf[1])
                ].value

            _, p_val = ss.kruskal(*samples.T)
            p_values[roi] = p_val

        return p_values

    def are_responses_significant(self):
        p_st = {}
        p_wsrt = {}
        for roi in range(self.data.n_roi):
            subset = self.responses[self.responses.roi_id == roi]

            # Sign test (implemented with binomial test)
            p_st[roi] = ss.binom_test(
                sum([1 for d in subset.subtracted if d > 0]),
                n=len(subset.subtracted),
                alternative="greater",
            )

            # Wilcoxon signed rank test
            _, p_wsrt[roi] = ss.wilcoxon(
                x=subset.subtracted,
                alternative="greater",
            )

        return p_st, p_wsrt

    def get_preferred_direction_all_rois(self):
        raise NotImplementedError("This method is not implemented yet")

    def _fit_two_dimensional_elliptical_gaussian(self):
        # as described by Priebe et al. 2006
        # add the variations added by Andermann et al. 2011 / 2013
        # calls _2d_gaussian
        # calls _get_response_map
        raise NotImplementedError("This method is not implemented yet")


class Gaussian2D:
    # a 2D gaussian function
    # also used by plotting functions
    def __init__(self):
        # different kinds of 2D gaussians:
        # - 2D gaussian
        # - 2D gaussian Andermann
        # - 2D gaussian Priebe
        raise NotImplementedError("This method is not implemented yet")


class ResponseMap:
    # also used by plotting functions
    def _get_response_map(self):
        # calls _get_preferred_direction
        raise NotImplementedError("This method is not implemented yet")

    def _get_preferred_direction(self):
        raise NotImplementedError("This method is not implemented yet")
