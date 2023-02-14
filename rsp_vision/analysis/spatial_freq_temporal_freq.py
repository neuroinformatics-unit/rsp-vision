import numpy as np
from utils import get_fps

from ..objects.enums import PhotonType
from ..objects.photon_data import PhotonData


class SF_TF:
    def __init__(
        self, data: PhotonData, config: dict, photon_type: PhotonType
    ):
        self.data = data
        self.photon_type = photon_type

        self.fps = get_fps(photon_type)

        self.padding_start = int(config["padding"][0])
        self.padding_end = int(config["padding"][1])

    def get_response_matrix_from_padding(self, day_idx=0, roi_idx=0):
        stimulus_idxs = self.data.signal[self.data.signal["stimulus_onset"]]
        self.n_frames_for_dispalay = (
            self.data.n_frames_per_trigger * self.data.n_triggers_per_stimulus
            + self.padding_end
        )
        full_matrix = np.zeros(
            (self.n_frames_for_dispalay, len(stimulus_idxs))
        )

        filtered_signal = self.data.signal[
            (self.data.signal["day_idx"] == day_idx)
            & (self.data.signal["roi_idx"] == roi_idx)
        ]
        for i, idx in enumerate(stimulus_idxs):
            full_matrix[:, i] = filtered_signal[
                idx - self.padding_start : idx + self.padding_end
            ]["signal"]

        response_matrix = np.mean(full_matrix[self.fps :, :], axis=0)
        baseline_matrix = np.mean(full_matrix[-2 * self.fps :, :], axis=0)
        return response_matrix - baseline_matrix

    def get_fit_parameters(self):
        # calls _fit_two_dimensional_elliptical_gaussian
        raise NotImplementedError("This method is not implemented yet")

    def responsiveness(self):
        raise NotImplementedError("This method is not implemented yet")

    def responsiveness_anova_window(self):
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
