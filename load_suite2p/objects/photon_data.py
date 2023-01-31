from ..analysis.remapping import Remapping
from ..objects.enums import PhotonType
from .data_raw import DataRaw


class PhotonData:
    def __init__(self, data_raw: DataRaw):

        self.screen_size = data_raw.stim[0]["screen_size"]

        self.mapper = Remapping(data_raw, PhotonType.TWO_PHOTON)
        self.signal = self.mapper.make_signal_dataframe(data_raw)
