from ..load.photon_data import PhotonData, PhotonOptions


class SparseNoise:
    def __init__(self, data: PhotonData, options: PhotonOptions):
        self.data = data
        self.options = options

    def responsiveness(self):
        raise NotImplementedError("This method is not implemented yet")
