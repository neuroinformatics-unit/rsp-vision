import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from rsp_vision.analysis.utils import ascii_array_to_string, get_fps
from rsp_vision.objects.data_raw import DataRaw
from rsp_vision.objects.enums import PhotonType


class PhotonData:
    """Class to remap the data from the raw data to a more usable format,
    such as a pandas dataframe.
    Its main output is a dataframe called `signal`, which contains all
    the information that would be useful for plotting.
    It does not include padding calculations, which I would like it
    to be done dynamically while browsing the data.
    """

    def __init__(
        self,
        data_raw: DataRaw,
        photon_type: PhotonType,
        config: dict,
        deactivate_checks=False,
    ):
        self.deactivate_checks = deactivate_checks
        self.photon_type = photon_type
        self.config = config
        self.fps = get_fps(self.photon_type, self.config)
        self.set_general_variables(data_raw)
        self.signal = self.get_signal_df(data_raw)
        logging.info(
            "Some of the data extracted:\n"
            + f"{self.signal[self.signal['stimulus_onset']].head()}"
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
        self.n_sessions = data_raw.frames.shape[0]
        self.n_roi = data_raw.frames[0].shape[0]
        self.n_frames_per_session = data_raw.frames[0].shape[1]
        self.day_stim = data_raw.day["stimulus"]  # seems useless
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

        self.n_all_triggers = data_raw.stim[0]["n_triggers"]
        self.n_session_boundary_baseline_triggers = int(
            data_raw.stim[0]["stimulus"]["n_baseline_triggers"]
        )

        self.n_stimulus_triggers_across_all_sessions = (
            self.n_all_triggers - 2 * self.n_session_boundary_baseline_triggers
        ) * self.n_sessions
        self.n_of_stimuli_across_all_sessions = int(
            self.n_stimulus_triggers_across_all_sessions
            / self.n_triggers_per_stimulus
        )

        self.calculations_to_find_start_frames()

    def calculations_to_find_start_frames(self):
        """Calculations to find the start frames of the stimuli,
        as in the matlab codebase.
        """
        self.n_frames_per_trigger = (
            self.n_frames_per_session / self.n_all_triggers
        )
        if (
            self.n_frames_per_trigger
            != self.config["trigger_interval_s"] * self.fps
        ):
            message = f"Frames per trigger from data: \
            {self.n_frames_per_trigger} and calculated from fps: \
            {self.config['trigger_interval_s'] * self.fps} are different"
            logging.error(message)
            raise RuntimeError(
                "Number of frames per trigger is wrong\n" + message
            )

        self.n_baseline_frames = (
            self.n_session_boundary_baseline_triggers
            * self.n_frames_per_trigger
        )
        self.n_stimulus_triggers_per_session = int(
            self.n_all_triggers - 2 * self.n_session_boundary_baseline_triggers
        )
        self.n_of_stimuli_per_session = int(
            self.n_stimulus_triggers_per_session / self.n_triggers_per_stimulus
        )
        self.stimulus_start_frames = self.get_stimulus_start_frames()

    def get_stimulus_start_frames(self):
        # I assume the signal has been cut in the generation of
        # this summary data in order to allign perfectly
        # It needs to be checked with trigger information

        frames_in_session = np.array(
            (
                self.n_baseline_frames
                + np.arange(0, self.n_of_stimuli_per_session)
                * self.n_triggers_per_stimulus
                * self.n_frames_per_trigger
                + 1
            ),
            dtype=int,
        )

        frames_all_sessions = (
            frames_in_session.reshape(-1, 1)
            + np.arange(0, self.n_sessions) * self.n_frames_per_session
        )
        frames_all_sessions = np.sort(frames_all_sessions.flatten(), axis=None)

        return frames_all_sessions

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
                        "frames_id": np.arange(
                            self.n_frames_per_session * session,
                            self.n_frames_per_session * (session + 1),
                        ),
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
        td = timedelta(seconds=(1 / get_fps(self.photon_type, self.config)))
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
                        signal_idx, self.n_of_stimuli_per_session
                    ),
                }
            )
            stimuli = pd.concat([stimuli, df])

        self.check_consistency_of_stimuli_df(stimuli)

        self.uniques = {
            "sf": stimuli.sf.unique(),
            "tf": stimuli.tf.unique(),
            "direction": stimuli.direction.unique(),
        }
        return stimuli

    def check_consistency_of_stimuli_df(self, stimuli):
        """
        Check the consistency of stimuli dataframe with the expected
        number of stimuli and their combinations.

        Parameters
        ----------
        stimuli : pandas DataFrame
            DataFrame containing stimuli information.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the number of stimuli in the input dataframe is not equal
            to the expected number of stimuli or if there are missing or
            duplicated combinations of stimuli.

        Notes
        -----
        The method first checks if the number of stimuli in the input
        dataframe is equal to the expected number of stimuli across all
        sessions. If not, it raises a RuntimeError and logs an error message.

        Then, it creates a pivot table of the stimuli dataframe based on
        the "sf", "tf", and "direction" columns and checks if the number
        of rows in the pivot table is equal to the expected number of stimuli
        combinations. If not, it raises a RuntimeError and logs an error
        message.

        Finally, it checks if the number of triggers per stimulus is
        consistent across all stimuli combinations in the pivot table.
        If not, it raises a RuntimeError and logs an error message.
        """
        if not self.deactivate_checks:
            if len(stimuli) != self.n_of_stimuli_across_all_sessions:
                logging.error(
                    f"Len of stimuli table: {len(stimuli)}, calculated "
                    + f"stimuli lenght: \
                    {self.n_of_stimuli_across_all_sessions}"
                )
                raise RuntimeError(
                    "Number of stimuli in raw_data.stim differs from the "
                    + "calculated amount of stimuli"
                )

            pivot_table = stimuli.pivot_table(
                index=["sf", "tf", "direction"], aggfunc="size"
            )

            if (
                len(pivot_table)
                != self.config["n_sf"]
                * self.config["n_tf"]
                * self.config["n_dir"]
            ):
                logging.error(f"Pivot table: {pivot_table}")
                logging.error(f"Pivot table length: {len(pivot_table)}")
                raise RuntimeError(
                    "Number of stimuli is not correct, some combinations are "
                    + "missing or duplicated"
                )

            if np.any(pivot_table.values != self.n_triggers_per_stimulus):
                logging.error(f"Pivot table: {pivot_table}")
                raise RuntimeError(
                    "Number of stimuli is not correct, some combinations are "
                    + "missing or duplicated"
                )

    def fill_up_with_stim_info(self, signal, stimuli):
        """Complete the signal dataframe with the stimulus information.

        Parameters
        ----------
        signal : pd.DataFrame
            Signal dataframe to be filled up with stimulus information
        stimuli : pd.DataFrame
            Dataframe with the stimuli information

        Returns
        -------
        signal : pd.DataFrame
            Final signal dataframe with stimulus information
        """

        # register only the stimulus onset
        for stimulus_index, start_frame in enumerate(
            self.stimulus_start_frames
        ):
            mask = signal["frames_id"] == start_frame
            signal_idxs = signal.index[mask]

            # starting frames and stimuli are alligned in source data
            stimulus = stimuli.iloc[stimulus_index]

            if len(signal_idxs) != self.n_roi and not self.deactivate_checks:
                raise RuntimeError(
                    f"Number of instances for stimulus {stimulus} is wrong."
                )

            # there would be the same start frame for each session for each roi
            signal.loc[mask, "sf"] = stimulus["sf"]
            signal.loc[mask, "tf"] = stimulus["tf"]
            signal.loc[mask, "direction"] = stimulus["direction"]
            signal.loc[mask, "stimulus_onset"] = True

        self.check_consistency_of_signal_df(signal)

        logging.info("Stimulus information added to signal dataframe")

        return signal

    def check_consistency_of_signal_df(self, signal):
        """Check the consistency of the signal dataframe with the
        expected number of stimuli.

        Parameters
        ----------
        signal : pd.DataFrame
            Signal dataframe with stimulus information

        Raises
        ------
        ValueError
            If the signal table was not populated correctly with
            stimulus information.
        """
        if not self.deactivate_checks:
            pivot_table = signal[signal.stimulus_onset].pivot_table(
                index=["sf", "tf", "direction"], aggfunc="size"
            )
            expected = self.n_triggers_per_stimulus * self.n_roi
            if not np.all(pivot_table == expected):
                raise ValueError(
                    f"Signal table was not populated correctly \
                    with stimulus information.\nPivot table:{pivot_table}, \
                    expercted:{expected}"
                )
