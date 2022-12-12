from .configurations import Config


class PhotonData:
    """Class to load the formatted data from suite2p and registers2p."""

    def __init__(self, data_raw: list, config: Config):

        self.response_matrix = self.get_response_matrix()
        self.preprocess(data_raw, config)
        self.reorder()

        self.f = None
        self.stim = None
        self.trig = None
        self.drift_order = None
        self.day = None
        self.is_cell = None

    def get_response_matrix(self):
        raise NotImplementedError("This method is not implemented yet")

    def preprocess(self, data_raw: list, config: Config):
        raise NotImplementedError("This method is not implemented yet")

    def reorder(self):
        raise NotImplementedError("This method is not implemented yet")