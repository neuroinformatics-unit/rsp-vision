from .data_raw import DataRaw
from .specifications import Specifications


class PhotonData:
    """Class to load the formatted data from suite2p and registers2p."""

    def __init__(self, data_raw: DataRaw, specs: Specifications):

        self.response_matrix = self.get_response_matrix()
        self.preprocess(data_raw, specs)
        self.reorder()

        self.f = None
        self.stim = None
        self.trig = None
        self.drift_order = None
        self.day = None
        self.is_cell = None

    def get_response_matrix(self):
        raise NotImplementedError("This method is not implemented yet")

    def preprocess(self, data_raw: DataRaw, specs: Specifications):
        raise NotImplementedError("This method is not implemented yet")

    def reorder(self):
        raise NotImplementedError("This method is not implemented yet")
