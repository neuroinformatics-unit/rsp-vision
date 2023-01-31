from ..analysis.remapping import Remapping
from ..objects.enums import PhotonType
from .data_raw import DataRaw


class PhotonData:
    def __init__(self, data_raw: DataRaw):

        # initialize valiable from data_raw that are straight forward
        self.is_cell = data_raw.is_cell
        self.day_roi = data_raw.day["roi"]
        self.day_roi_label = data_raw.day["roi_label"]

        # hard coded values
        self.padding = [25, 50]
        self.drift_order = 2
        self.dim_name = [
            "frame",
            "trial",
            "orientation",
            "spatial_frequency",
            "temporal_frequency",
            "roi",
        ]
        self.screen_size = data_raw.stim[0]["screen_size"]

        self.mapper = Remapping(data_raw, PhotonType.TWO_PHOTON)
        self.signal = self.mapper.make_signal_dataframe(data_raw)
