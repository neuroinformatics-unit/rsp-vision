import logging

import numpy as np

from ..analysis.utils import ascii_array_to_string
from ..objects.data_raw import DataRaw


class ReorganizeData:
    def __init__(self, padding):
        self.padding = padding

    def extract_arrays_from_raw_data(self, data_raw: DataRaw):
        F, day_stim = self.get_F_and_day_stim(data_raw)

        grey_idx, drift_idx, static_idx = None, None, None  # TODO

        return F, day_stim, grey_idx, drift_idx, static_idx

    def get_F_and_day_stim(self, data_raw: DataRaw):
        # some stim values
        grey_or_static = ascii_array_to_string(
            data_raw.stim[0]["stimulus"]["grey_or_static"]
        )
        n_triggers_per_stimulus = (
            3
            if grey_or_static
            in ["grey_static_drift", "grey_static_drift_switch"]
            else 2
        )
        n_baseline_trigger = int(
            data_raw.stim[0]["stimulus"]["n_baseline_triggers"]
        )
        n_trigger = data_raw.stim[0]["n_triggers"]

        # define shape of f_store
        n_roi = self.get_n_roi(data_raw.frames)
        n_sessions = self.get_n_sessions(data_raw.frames)
        total_n_stimulus_triggers = self.get_total_n_stimulus_triggers(
            n_trigger, n_baseline_trigger, n_sessions
        )
        n_frames_per_session = self.get_n_frames_per_session(data_raw.frames)
        n_frames_per_trigger = self.get_n_frames_per_trigger(
            n_frames_per_session, n_trigger
        )
        total_n_of_stimuli = self.get_total_n_of_stimuli(
            total_n_stimulus_triggers, n_triggers_per_stimulus
        )
        n_frames_for_display = self.get_frames_for_display(
            n_frames_per_trigger,
            self.padding,
            n_triggers_per_stimulus,
        )
        f_store = np.empty(
            (int(n_frames_for_display), int(total_n_of_stimuli), int(n_roi))
        )

        # find other variables and loop over sessions to fill f_store
        n_baseline_frames = n_baseline_trigger * n_frames_per_trigger
        n_stimulus_triggers = self.get_stimulus_triggers(
            n_trigger, n_baseline_trigger
        )
        inner_total_n_of_stimuli = self.get_total_n_of_stimuli(
            n_stimulus_triggers, n_triggers_per_stimulus
        )
        stimulus_start_frames = self.get_stimulus_start_frames(
            n_baseline_frames,
            inner_total_n_of_stimuli,
            n_triggers_per_stimulus,
            n_frames_per_trigger,
        )
        common_idx = self.get_common_idx(
            self.padding, n_triggers_per_stimulus, n_frames_per_trigger
        )
        roi_offset = self.get_roi_offset(n_roi, n_frames_per_session)
        frames_idx = self.get_frames_idx(
            common_idx, stimulus_start_frames, roi_offset
        )

        for k in range(n_sessions):
            logging.info(
                "Loading session " + str(k) + " of " + str(n_sessions)
            )

            f_reshape = self.get_f_reshape(
                data_raw.frames, k, frames_idx
            )  # To be improved
            f_store[k] = f_reshape  # Placeholder

        day_stim = None  # TODO

        return f_store, day_stim

    @staticmethod
    def get_n_roi(f: np.ndarray):
        return f[0].shape[0]

    @staticmethod
    def get_n_frames_per_session(f: np.ndarray):
        return f[0].shape[1]

    @staticmethod
    def get_n_sessions(f: np.ndarray):
        return f.shape[0]

    @staticmethod
    def get_total_n_stimulus_triggers(
        n_trigger, n_baseline_trigger, n_sessions
    ):
        return (n_trigger - 2 * n_baseline_trigger) * n_sessions

    @staticmethod
    def get_n_frames_per_trigger(n_frames_per_session: int, n_trigger: int):
        return n_frames_per_session / n_trigger

    @staticmethod
    def get_total_n_of_stimuli(
        total_n_stimulus_triggers: int, n_triggers_per_stimulus: int
    ):
        return total_n_stimulus_triggers / n_triggers_per_stimulus

    @staticmethod
    def get_frames_for_display(
        n_frames_per_trigger, padding, n_triggers_per_stimulus
    ):
        return n_triggers_per_stimulus * n_frames_per_trigger + np.sum(padding)

    @staticmethod
    def get_stimulus_triggers(n_trigger, n_baseline_trigger):
        return int(n_trigger - 2 * n_baseline_trigger)

    @staticmethod
    def get_stimulus_start_frames(
        n_baseline_frames,
        total_n_of_stimuli,
        n_triggers_per_stimulus,
        n_frames_per_trigger,
    ):
        return (
            n_baseline_frames
            + np.arange(0, total_n_of_stimuli)
            * n_triggers_per_stimulus
            * n_frames_per_trigger
            + 1
        )

    @staticmethod
    def get_frames_idx(common_idx, stimulus_start_frames, roi_offset):

        inner_array = common_idx + stimulus_start_frames[:, np.newaxis]
        return np.array([x + inner_array for x in roi_offset]).reshape(
            300, 48, 11
        )

    @staticmethod
    def get_common_idx(padding, n_triggers_per_stimulus, n_frames_per_trigger):
        return np.arange(
            -padding[0],
            n_triggers_per_stimulus * n_frames_per_trigger + padding[1],
        ).T

    @staticmethod
    def get_roi_offset(n_roi, n_frames_per_session):
        return np.arange(0, n_roi) * n_frames_per_session

    @staticmethod
    def get_f_reshape(f, k, frames_idx):
        total_matrix = []
        for n_roi in range(frames_idx.shape[2]):
            block = []
            for n_baseline_frames in range(frames_idx.shape[0]):
                for total_n_of_stimuli in range(frames_idx.shape[1]):
                    try:
                        column = [
                            f[k][n_roi][int(x) - 1]
                            for x in frames_idx[total_n_of_stimuli][
                                n_baseline_frames
                            ]
                        ]
                        block.append(column)
                    except IndexError:
                        pass

            total_matrix.append(block)

        array = np.array(total_matrix)
        # simplify code above

        return array.reshape(300, 48, 11)
