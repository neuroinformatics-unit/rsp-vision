from tests.test_integration.generate_mock_data import (
    n_roi,
)


def test_set_general_variables(one_day_objects, multiple_days_objects):
    for data, v, _ in [one_day_objects, multiple_days_objects]:
        assert data.total_n_days == v.n_days
        assert data.n_sessions == v.n_sessions
        assert data.n_roi == n_roi
        assert data.n_frames_per_session == v.len_session
        assert data.n_of_stimuli_per_session == v.n_stim / v.n_sessions
        assert data.stimulus_start_frames.shape[0] == v.n_stim
        assert data.n_of_stimuli_across_all_sessions == (v.n_stim)


def test_make_signal_dataframe(one_day_objects, multiple_days_objects):
    for data, v, data_raw in [one_day_objects, multiple_days_objects]:
        signal = data.make_signal_dataframe(data_raw)

        number_of_columns = 12  # refers to the shape of the DataFrame

        assert signal.shape == (
            v.len_session * v.n_sessions * n_roi,
            number_of_columns,
        )


def test_get_stimuli(one_day_objects, multiple_days_objects):
    for data, v, data_raw in [one_day_objects, multiple_days_objects]:
        stimuli = data.get_stimuli(data_raw)

        assert stimuli.shape == (v.n_stim, 4)


def test_fill_up_with_stim_info(one_day_objects, multiple_days_objects):
    for data, v, data_raw in [one_day_objects, multiple_days_objects]:
        signal = data.make_signal_dataframe(data_raw)
        stimuli = data.get_stimuli(data_raw)
        signal = data.fill_up_with_stim_info(signal, stimuli)

        frames = set(signal[signal["stimulus_onset"]].frames_id)

        assert frames == set(data.stimulus_start_frames)
