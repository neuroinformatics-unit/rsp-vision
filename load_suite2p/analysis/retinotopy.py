from ..load.formatted_data import PhotonData, PhotonOptions


class Retinotopy:
    def __init__(self, data: PhotonData, options: PhotonOptions):
        self.data = data
        self.options = options

    def responsiveness(self):
        """Computes the responsiveness of each cell to the stimulus."""
        raise NotImplementedError("Not implemented yet")

    def response_magnitude(self):
        """Computes the response magnitude of each cell to the stimulus."""
        raise NotImplementedError("Not implemented yet")

    def get_retinotopy_map(self):
        """Computes the retinotopy map of each cell."""
        raise NotImplementedError("Not implemented yet")
