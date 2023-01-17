# other mocks
import numpy as np

from load_suite2p.analysis.reorganize_data import ReorganizeData


def create_data_raw_mock():
    f = np.ones((18), dtype=object)
    f[0] = np.zeros((11, 11400))

    x = {
        "stimulus": {
            "n_baseline_triggers": np.array([4]),
        },
        "n_triggers": np.array([152]),
        "screen_size": "screen_size",
    }
    stim = np.repeat(x, 18, axis=0)

    data = {
        "day": {
            "roi": "roi",
            "roi_label": "roi_label",
            "stimulus": "stimulus",
        },
        "f": f,
        "imaging": "imaging",
        "is_cell": "is_cell",
        "r_neu": "r_neu",
        "stim": stim,
        "trig": "trig",
    }
    return data


def test_get_n_sessions():
    data_raw = create_data_raw_mock()
    assert ReorganizeData.get_n_sessions(data_raw["f"]) == 18


def test_get_n_roi():
    data_raw = create_data_raw_mock()
    assert ReorganizeData.get_n_roi(data_raw["f"]) == 11


def test_get_total_n_stimulus_triggers():
    n_trigger, n_baseline_trigger, n_sessions = 152, 4, 18
    assert (
        ReorganizeData.get_total_n_stimulus_triggers(
            n_trigger, n_baseline_trigger, n_sessions
        )
        == 2592
    )


def test_get_n_frames_per_trigger():
    n_frames_per_session, n_trigger = 11400, 152
    assert (
        ReorganizeData.get_n_frames_per_trigger(
            n_frames_per_session, n_trigger
        )
        == 75
    )


def test_get_total_n_of_stimuli():
    assert (
        ReorganizeData.get_total_n_of_stimuli(
            total_n_stimulus_triggers=2592, n_triggers_per_stimulus=3
        )
        == 864
    )


def test_get_frames_for_display():
    assert (
        ReorganizeData.get_frames_for_display(
            n_frames_per_trigger=75,
            padding=[25, 50],
            n_triggers_per_stimulus=3,
        )
        == 300
    )


def test_get_stimulus_start_frames():
    my_range = 300 + np.arange(0, 48) * 3 * 75 + 1
    assert np.all(
        ReorganizeData.get_stimulus_start_frames(
            n_baseline_frames=300,
            total_n_of_stimuli=48,
            n_triggers_per_stimulus=3,
            n_frames_per_trigger=75,
        )
        == my_range
    )


def test_get_common_idx():
    array = np.arange(-25, 275)
    padding = [25, 50]
    assert np.all(ReorganizeData.get_common_idx(padding, 3, 75) == array)


def test_get_frames_idx_shape():
    common_idx = np.arange(-25, 275)
    start_frames = 300 + np.arange(0, 48) * 3 * 75 + 1
    roi_offset = np.arange(0, 125400, 11400)

    assert ReorganizeData.get_frames_idx(
        common_idx, start_frames, roi_offset
    ).shape == (300, 48, 11)


def test_get_frames_idx_specific_value():
    common_idx = np.arange(-25, 275)
    start_frames = 300 + np.arange(0, 48) * 3 * 75 + 1
    roi_offset = np.arange(0, 125400, 11400)
    assert (
        ReorganizeData.get_frames_idx(common_idx, start_frames, roi_offset)[
            4, 4, 4
        ]
        == 1911
    )


def test_get_stimulus_triggers():
    n_trigger, n_baseline_trigger = 152, 4
    assert np.all(
        ReorganizeData.get_stimulus_triggers(n_trigger, n_baseline_trigger)
        == 144
    )


def test_get_n_frames_per_session():
    data_raw = create_data_raw_mock()
    assert np.all(
        ReorganizeData.get_n_frames_per_session(data_raw["f"]) == 11400
    )


def test_get_roi_offset():
    array = np.arange(0, 125400, 11400)
    assert np.all(ReorganizeData.get_roi_offset(11, 11400) == array)
