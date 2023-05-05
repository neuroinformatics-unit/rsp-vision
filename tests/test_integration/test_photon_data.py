def test_set_general_variables(experimental_variables, photon_data):
    n_sessions, n_roi, len_session, n_stim, _, _ = experimental_variables

    assert photon_data.n_sessions == n_sessions
    assert photon_data.n_roi == n_roi
    assert photon_data.n_frames_per_session == len_session
    assert photon_data.n_of_stimuli_per_session == n_stim / 2
    assert (
        photon_data.stimulus_start_frames.shape[0] == n_sessions * n_stim / 2
    )


def test_make_signal_dataframe(photon_data, data_raw, experimental_variables):
    signal = photon_data.make_signal_dataframe(data_raw)
    n_sessions, n_roi, len_session, _, _, _ = experimental_variables

    number_of_columns = 12

    assert signal.shape == (
        len_session * n_sessions * n_roi,
        number_of_columns,
    )


def test_get_stimuli(photon_data, data_raw, experimental_variables):
    _, _, _, n_stim, _, _ = experimental_variables

    stimuli = photon_data.get_stimuli(data_raw)

    assert stimuli.shape == (n_stim, 4)


def test_fill_up_with_stim_info(photon_data, data_raw):
    signal = photon_data.make_signal_dataframe(data_raw)
    stimuli = photon_data.get_stimuli(data_raw)
    signal = photon_data.fill_up_with_stim_info(signal, stimuli)

    frames = set(signal[signal["stimulus_onset"]].frames_id)

    assert frames == set(photon_data.stimulus_start_frames)
