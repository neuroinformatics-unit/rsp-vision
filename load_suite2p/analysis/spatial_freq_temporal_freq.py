from ..load.formatted_data import PhotonData, PhotonOptions


class SpatialFrequencyTemporalFrequency:
    def __init__(self, data: PhotonData, options: PhotonOptions):
        self.data = data
        self.options = options

    def get_fit_parameters(self):
        raise NotImplementedError("This method is not implemented yet")

    def responsiveness_anova_window(self):
        raise NotImplementedError("This method is not implemented yet")

    def get_preferred_direction_all_rois(self):
        raise NotImplementedError("This method is not implemented yet")
