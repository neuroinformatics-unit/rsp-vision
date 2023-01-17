from ..analysis.reorganize_data import ReorganizeData
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

        # to be caluculated
        organizer = ReorganizeData(self.padding)
        (
            self.F,
            self.day_stim,
            self.grey_idx,
            self.drift_idx,
            self.static_idx,
        ) = organizer.extract_arrays_from_raw_data(data_raw)
