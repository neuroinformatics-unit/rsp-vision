import logging

import numpy as np

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
        window_start_response = stimulus_idxs + (
            self.data.n_frames_per_trigger * response_start
        )
        window_end_response = stimulus_idxs + (
            self.data.n_frames_per_trigger * (response_start + 1)
        )
        window_mask_response = np.vstack(
            [
                np.arange(start, end)
                for start, end in zip(
                    window_start_response, window_end_response
                )
            ]
        )

        window_start_baseline = stimulus_idxs + (
            self.data.n_frames_per_trigger * baseline_start
        )
        window_end_baseline = stimulus_idxs + (
            self.data.n_frames_per_trigger * (baseline_start + 1)
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

        self.only_stim_onset = self.adapted_signal[
            self.adapted_signal["stimulus_onset"]
        ]
        logging.info(f"Adapted signal dataframe:{self.only_stim_onset.head()}")

    def get_fit_parameters(self):
        # calls _fit_two_dimensional_elliptical_gaussian
        raise NotImplementedError("This method is not implemented yet")

    def responsiveness(self):
        self.responsiveness_anova()
        raise NotImplementedError("This method is not implemented yet")

    def responsiveness_anova(self):
        raise NotImplementedError("This method is not implemented yet")

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
