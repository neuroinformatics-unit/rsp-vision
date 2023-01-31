import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from ..analysis.utils import ascii_array_to_string
from ..objects.data_raw import DataRaw
from ..objects.enums import PhotonType


class Remapping:
    def __init__(self, data_raw: DataRaw, photon_type: PhotonType):
        self.photon_type = photon_type
        self.set_general_variables(data_raw)

    def get_signal_df(self, data_raw: DataRaw):
        signal = self.make_signal_dataframe(data_raw)
        stimuli = self.get_stimuli(data_raw)
        signal = self.fill_up_with_stim_info(signal, stimuli)

        return signal

    def set_general_variables(self, data_raw):
        self.n_sessions = data_raw.frames.shape[0]
        self.n_roi = data_raw.frames[0].shape[0]
        self.n_frames_per_session = data_raw.frames[0].shape[1]
        self.day_stim = data_raw.day["stimulus"]
        self.grey_or_static = ascii_array_to_string(
            data_raw.stim[0]["stimulus"]["grey_or_static"]
        )
        if self.grey_or_static in [
            "grey_static_drift",
            "grey_static_drift_switch",
        ]:
            self.is_gray = True
            self.n_triggers_per_stimulus = 3
        else:
            self.is_gray = False
            self.n_triggers_per_stimulus = 2

        self.n_trigger = data_raw.stim[0]["n_triggers"]
        self.n_baseline_trigger = int(
            data_raw.stim[0]["stimulus"]["n_baseline_triggers"]
        )

        self.total_n_stimulus_triggers = (
            self.n_trigger - 2 * self.n_baseline_trigger
        ) * self.n_sessions
        self.total_n_of_stimuli = (
            self.total_n_stimulus_triggers / self.n_triggers_per_stimulus
        )

        self.calculations_to_find_start_frames()

    def calculations_to_find_start_frames(self):
        # calculations as in the matlab codebase
        self.n_frames_per_trigger = self.n_frames_per_session / self.n_trigger
        self.n_baseline_frames = (
            self.n_baseline_trigger * self.n_frames_per_trigger
        )
        self.n_stimulus_triggers = int(
            self.n_trigger - 2 * self.n_baseline_trigger
        )
        self.inner_total_n_of_stimuli = (
            self.n_stimulus_triggers / self.n_triggers_per_stimulus
        )
        self.stimulus_start_frames = self.get_stimulus_start_frames()

    def get_stimulus_start_frames(self):
        # I assume the signal has been cut in the generation of
        # this summary data in order to allign perfectly
        # It needs to be checked with trigger information
        return np.array(
            (
                self.n_baseline_frames
                + np.arange(0, self.total_n_of_stimuli)
                * self.n_triggers_per_stimulus
                * self.n_frames_per_trigger
                + 1
            ),
            dtype=int,
        )

    def make_signal_dataframe(self, data_raw):

        signal = pd.DataFrame(
            columns=[
                "day",
                "timedelta from beginning",
                "frames_id",
                "signal",
                "roi_id",
                "session_id",
                "sf",
                "tf",
                "direction",
                "stimulus_onset",
            ]
        )
        signal = signal.astype(
            {
                "day": "int32",
                "timedelta from beginning": "timedelta64[ns]",
                "frames_id": "int32",
                "signal": "float32",
                "roi_id": "int32",
                "session_id": "int32",
                "sf": "float32",
                "tf": "float32",
                "direction": "float32",
                "stimulus_onset": "bool",
            }
        )

        for session in range(self.n_sessions):
            for roi in range(self.n_roi):
                df = pd.DataFrame(
                    {
                        "day": np.repeat(
                            self.day_stim[0], self.n_frames_per_session
                        ),
                        "timedelta from beginning": self.get_timing_array(),
                        "frames_id": np.arange(0, self.n_frames_per_session),
                        "signal": data_raw.frames[session][roi, :],
                        "roi_id": np.repeat(roi, self.n_frames_per_session),
                        "session_id": np.repeat(
                            session, self.n_frames_per_session
                        ),
                        "sf": np.repeat(np.nan, self.n_frames_per_session),
                        "tf": np.repeat(np.nan, self.n_frames_per_session),
                        "direction": np.repeat(
                            np.nan, self.n_frames_per_session
                        ),
                        "stimulus_onset": np.repeat(
                            False, self.n_frames_per_session
                        ),
                    }
                )

                signal = pd.concat([signal, df], ignore_index=True)

        logging.info("Signal dataframe created")

        return signal

    def get_timing_array(self):
        td = timedelta(seconds=(1 / self.get_fps()))
        return np.cumsum(np.repeat(td, self.n_frames_per_session))

    def get_fps(self):
        if self.photon_type == PhotonType.TWO_PHOTON:
            return 30
        elif self.photon_type == PhotonType.THREE_PHOTON:
            return 15
        else:
            raise NotImplementedError(
                "Unknown number of frames per second for this photon type"
            )

    def get_stimuli(self, data_raw):
        # dataframe of ordered stimuli to be mapped to signal table
        stimuli = pd.DataFrame(columns=["sf", "tf", "direction", "session"])
        for signal_idx in range(self.n_sessions):
            df = pd.DataFrame(
                {
                    "sf": data_raw.stim[signal_idx]["stimulus"][
                        "cycles_per_visual_degree"
                    ],
                    "tf": data_raw.stim[signal_idx]["stimulus"][
                        "cycles_per_second"
                    ],
                    "direction": data_raw.stim[signal_idx]["stimulus"][
                        "directions"
                    ],
                    "session": np.repeat(
                        signal_idx, self.inner_total_n_of_stimuli
                    ),
                }
            )
            stimuli = pd.concat([stimuli, df])
        return stimuli

    def fill_up_with_stim_info(self, signal, stimuli):
        # register only the stimulus onset
        for k, start_frame in enumerate(self.stimulus_start_frames):
            signal_idxs = signal.index[
                signal["frames_id"] == start_frame
            ].tolist()
            stimulus = stimuli.iloc[k]
            # there would be the same start frame for each session for each roi
            for i, signal_idx in enumerate(signal_idxs):
                signal.iloc[
                    signal_idx, signal.columns.get_loc("sf")
                ] = stimulus["sf"]
                signal.iloc[
                    signal_idx, signal.columns.get_loc("tf")
                ] = stimulus["tf"]
                signal.iloc[
                    signal_idx, signal.columns.get_loc("direction")
                ] = stimulus["direction"]
                signal.iloc[
                    signal_idx, signal.columns.get_loc("stimulus_onset")
                ] = True

        logging.info("Stimulus information added to signal dataframe")
        return signal
