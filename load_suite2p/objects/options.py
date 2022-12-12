class Options:
    """Class with the options to analize data from suite2p and registers2p."""

    def __init__(self, config: dict):
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
