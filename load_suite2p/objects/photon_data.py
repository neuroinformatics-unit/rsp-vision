import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..analysis.utils import ascii_array_to_string, get_fps
from ..objects.data_raw import DataRaw
from ..objects.enums import PhotonType


class PhotonData:
    """Class to remap the data from the raw data to a more usable format,
    such as a pandas dataframe.
    Its main output is a dataframe called `signal`, which contains all
    the information that would be useful for plotting.
    It does not include padding calculations, which I would like it
    to be done dynamically while browsing the data.
    """

    def __init__(self, data_raw: DataRaw, photon_type: PhotonType):
        self.photon_type = photon_type
        self.set_general_variables(data_raw)
        self.signal = self.get_signal_df(data_raw)
        logging.info(
            "Some of the data extracted:"
            + f"{self.signal[self.signal['stimulus_onset'] == True].head()}"
        )

    def get_signal_df(self, data_raw: DataRaw):
        """The main function of this class, which returns the final
        `signal` dataframe. It will contain the following columns:
        * day (int) - the day of the experiment (1, 2, 3 or 4)
        * time from beginning (datetime) - the time from the beginning
            of the experiment
        * frames_id (int) - the frame id
        * signal (float) - the signal value
        * roi_id (int) - the roi id
        * session_id (int) - the session id
        * sf (float) - the spatial frequency of the stimulus
        * tf (float) - the temporal frequency of the stimulus
        * direction (float) - the direction of the stimulus
        * stimulus_onset (bool) - whether the frame is a stimulus onset


        Parameters
        ----------
        data_raw : DataRaw
            The raw data object from which the data will be extracted

        Returns
        -------
        signal : pd.DataFrame
            The final dataframe
        """
        signal = self.make_signal_dataframe(data_raw)
        stimuli = self.get_stimuli(data_raw)
        signal = self.fill_up_with_stim_info(signal, stimuli)

        return signal

    def set_general_variables(self, data_raw):
        """Set the general variables that will be used in the class,
        mostly using the same calculations as in the matlab codebase.

        Parameters
        ----------
        data_raw : DataRaw
            The raw data object from which the data will be extracted
        """
        self.screen_size = data_raw.stim[0]["screen_size"]
        self.is_cell = data_raw.is_cell
        self.day_roi = data_raw.day["roi"]
        self.day_roi_label = data_raw.day["roi_label"]
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
        """Calculations to find the start frames of the stimuli,
        as in the matlab codebase.
        """
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
        """Make the signal dataframe, which will be filled up with
        the stimulus information later on.

        Parameters
        ----------
        data_raw : DataRaw
            The raw data object from which the data will be extracted

        Returns
        -------
        signal : pd.DataFrame
            Initialized dataframe with only partial information
        """

        signal = pd.DataFrame(
            columns=[
                "day",
                "time from beginning",
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
                "time from beginning": "datetime64[ns]",
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
                        "time from beginning": self.get_timing_array(),
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
        """Get the timing array for the signal dataframe calculating it
        from the number of frames per session given a photon type.

        Returns
        -------
        np.array(datetime)
            An array of datetime objects corresponding to the frames
        """
        td = timedelta(seconds=(1 / get_fps(self.photon_type)))
        cumulative_sum = np.cumsum(np.repeat(td, self.n_frames_per_session))
        return datetime.today() + cumulative_sum

    def get_stimuli(self, data_raw):
        """Get the stimuli dataframe from the raw data object.

        Parameters
        ----------
        data_raw : DataRaw
            The raw data object from which the data will be extracted

        Returns
        -------
        stimuli : pd.DataFrame
            Dataframe with the stimuli information
        """
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
        """Complete the signal dataframe with the stimulus information.

        Parameters
        ----------
        signal : pd.DataFrame
            Signal dataframe to be filled up with stimulus information
        stimuli : _type_
            Dataframe with the stimuli information

        Returns
        -------
        signal : pd.DataFrame
            Final signal dataframe with stimulus information
        """
        # register only the stimulus onset
        for k, start_frame in enumerate(self.stimulus_start_frames):
            signal_idxs = signal.index[
                signal["frames_id"] == start_frame
            ].tolist()

            # starting frames and stimuli are alligned in source data
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
