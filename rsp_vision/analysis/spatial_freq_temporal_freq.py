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

        self.calculate_mean_response_and_baseline()

    def calculate_mean_response_and_baseline(
        self,
    ):
        stimulus_idxs = self.data.signal[
            self.data.signal["stimulus_onset"]
        ].index

        self.adapted_signal = self.data.signal
        self.adapted_signal["mean_response"] = np.nan
        self.adapted_signal["mean_baseline"] = np.nan

        baseline_start = 0
        if self.data.n_triggers_per_stimulus == 3:
            response_start = 2
            if self.data.config["baseline"] == "static":
                baseline_start = 1
        if self.data.n_triggers_per_stimulus == 2:
            response_start = 1

        logging.info("Start to edit the signal dataframe...")

        # Identify the rows in the window of each stimulus onset
        window_start_response = (
            stimulus_idxs
            + (self.data.n_frames_per_trigger * response_start)
            + (self.fps * 0.5)  # ignore first 0.5s
            - 1
        )
        window_end_response = (
            stimulus_idxs
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
            stimulus_idxs
            + (self.data.n_frames_per_trigger * baseline_start)
            + (self.fps * 1.5)  # ignore first 1.5s
            - 1
        )
        window_end_baseline = (
            stimulus_idxs
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

        mean_response_signal = [
            np.mean(
                self.adapted_signal.iloc[
                    window_mask_response[i]
                ].signal.values,
                axis=0,
            )
            for i in range(len(window_mask_response))
        ]
        mean_baseline_signal = [
            np.mean(
                self.adapted_signal.iloc[
                    window_mask_baseline[i]
                ].signal.values,
                axis=0,
            )
            for i in range(len(window_mask_baseline))
        ]

        self.adapted_signal.loc[
            stimulus_idxs, "mean_response"
        ] = mean_response_signal
        self.adapted_signal.loc[
            stimulus_idxs, "mean_baseline"
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

            _sf = self.data.uniques["sf"]
            _tf = self.data.uniques["tf"]
            sf_tf_combinations = np.array(np.meshgrid(_sf, _tf)).T.reshape(
                -1, 2
            )
            _dir = self.data.uniques["direction"]

            samples = np.zeros(
                (
                    len(_dir) * self.data.n_triggers_per_stimulus,
                    len(sf_tf_combinations),
                )
            )

            for i, sf_tf in enumerate(sf_tf_combinations):
                samples[:, i] = melted[
                    (melted.sf == sf_tf[0]) & (melted.tf == sf_tf[1])
                ].value

            _, p_val = ss.kruskal(*samples)
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
                x=subset.mean_response,
                y=subset.mean_baseline,
                alternative="less",
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
