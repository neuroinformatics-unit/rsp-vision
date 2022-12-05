from ..load.photon_data import PhotonData, PhotonOptions


class SpatialFrequencyTemporalFrequency:
    def __init__(self, data: PhotonData, options: PhotonOptions):
        self.data = data
        self.options = options

    def get_fit_parameters(self):
        # calls _fit_two_dimensional_elliptical_gaussian
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

    def _2d_gaussian(self):
        raise NotImplementedError("This method is not implemented yet")

    def _get_response_map(self):
        # calls _get_response_map
        raise NotImplementedError("This method is not implemented yet")

    def _get_preferred_direction(self):
        raise NotImplementedError("This method is not implemented yet")
