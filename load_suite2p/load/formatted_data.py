from .folder_naming_specs import FolderNamingSpecs


class PhotonData:
    """Class to load the formatted data from suite2p and registers2p."""

    def __init__(self, file_name: str, config: dict):
        self.file_name = file_name
        self.file_specs = self.get_FileNamingSpecs(config)
        self.response_matrix = self.get_response_matrix()

        # Attributes to evaluate:
        # self.F
        # self.type
        # self.stim_idx
        # self.dim_size
        # self.dim_value
        # self.day_roi
        # self.n_roi

    def get_FileNamingSpecs(self, config) -> FolderNamingSpecs:
        return FolderNamingSpecs(self.file_name, config)

    def get_response_matrix(self):
        raise NotImplementedError("This method is not implemented yet")


class PhotonOptions:
    """Class with the options to analize data from suite2p and registers2p."""

    def __init__(self):
        self.response = self.ResponseOptions()
        self.fitting = self.FittingOptions()
        # Attributes to evaluate:
        # self.trials
        # self.days_to_analyze
        # self.directions
        # self.colour
        pass

    class ResponseOptions:
        def __init__(self):
            # Attributes to evaluate:
            # self.baseline_frames
            # self.frames
            # self.peak_window
            # self.peak_enable
            # self.measure
            # self.trials
            # self.average_using
            # self.subtract_baseline
            # self.response_frames
            # self.p_threashold
            # self.use_magnitude
            # self.magnitude_threshold
            pass

    class FittingOptions:
        def __init__(self):
            # Attributes to evaluate:
            # self.lower_bound
            # self.start_cond
            # self.upper_bound
            # self.jitter
            # self.n_starting_points
            pass

    class SftfMapOptions:
        def __init__(self):
            # Attributes to evaluate:
            # self.use_pref_dir
            pass
