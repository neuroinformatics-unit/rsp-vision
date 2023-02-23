import logging

import numpy as np

from ..objects.enums import PhotonType
from ..objects.photon_data import PhotonData
from .utils import get_fps


class SF_TF:
    def __init__(
        self, data: PhotonData, config: dict, photon_type: PhotonType
    ):
        self.data = data
        self.photon_type = photon_type

        self.fps = get_fps(photon_type, config)

        self.padding_start = int(config["padding"][0])
        self.padding_end = int(config["padding"][1])

        self.get_response_matrix_from_padding()

    def get_response_matrix_from_padding(
        self,
    ):
        stimulus_idxs = self.data.signal[
            self.data.signal["stimulus_onset"]
        ].index
        self.n_frames_for_dispalay = int(
            self.data.n_frames_per_trigger * self.data.n_triggers_per_stimulus
            + self.padding_start
            + self.padding_end
        )

        self.adapted_signal = self.data.signal
        self.adapted_signal["mean_response"] = np.nan
        self.adapted_signal["mean_baseline"] = np.nan

        logging.info("Starting to edit the signal daataframe...")

        for idx in stimulus_idxs:
            id = idx - self.padding_start

            self.adapted_signal.loc[
                idx, "mean_response"
            ] = self.adapted_signal[
                (self.adapted_signal.index >= id + self.fps)
                & (self.adapted_signal.index < id + self.n_frames_for_dispalay)
            ][
                "signal"
            ].mean()

            self.adapted_signal.loc[
                idx, "mean_baseline"
            ] = self.adapted_signal[
                (
                    self.adapted_signal.index
                    >= id + self.n_frames_for_dispalay - (2 * self.fps)
                )
                & (self.adapted_signal.index < id + self.n_frames_for_dispalay)
            ][
                "signal"
            ].mean()

        self.adapted_signal["subtracted"] = (
            self.adapted_signal["mean_response"]
            - self.adapted_signal["mean_baseline"]
        )

        view = self.adapted_signal[self.adapted_signal["stimulus_onset"]]
        logging.info(f"Adapted signal dataframe:{view.head()}")

        # in the last trigger baseline is not calculated
        # because indexes go beyond the length of the dataframe

    def get_fit_parameters(self):
        # calls _fit_two_dimensional_elliptical_gaussian
        raise NotImplementedError("This method is not implemented yet")

    def responsiveness(self, rois: list):
        raise NotImplementedError("This method is not implemented yet")

    def responsiveness_anova(self):
        # perform non-parametric one way anova and
        # return the p-value for all combos across sesssions

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
