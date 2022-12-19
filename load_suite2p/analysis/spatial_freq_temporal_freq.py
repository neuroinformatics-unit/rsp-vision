from ..objects.photon_data import PhotonData
from ..objects.specifications import Specifications


class SF_TF:
    def __init__(self, data: PhotonData, specs: Specifications):
        self.data = data
        self.config = specs

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
