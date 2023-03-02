import numpy as np
import pytest

from rsp_vision.analysis.utils import get_fps
from rsp_vision.objects.data_raw import DataRaw
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData


@pytest.fixture
def get_variables():
    n_sessions = 2
    n_roi = 5
    n_stim = 8
    n_baseline_triggers = 4
    n_triggers_per_stim = 3
    n_frames_per_trigger = 75

    len_session = int(
        (2 * n_baseline_triggers + n_stim / n_sessions * n_triggers_per_stim)
        * n_frames_per_trigger
    )

    return (
        n_sessions,
        n_roi,
        len_session,
        n_stim,
        n_baseline_triggers,
        n_triggers_per_stim,
    )


@pytest.fixture
def get_config():
    yield {
        "fps_two_photon": 30,
        "trigger_interval_s": 2.5,
        "n_sf": 6,
        "n_tf": 6,
        "n_dir": 8,
    }


@pytest.fixture
def get_data_raw_object(get_variables):
    (
        n_sessions,
        n_roi,
        len_session,
        n_stim,
        n_baseline_triggers,
        n_triggers_per_stim,
    ) = get_variables

    data = {
        "day": {
            "roi": "roi",
            "roi_label": "roi_label",
            "stimulus": "stimulus",
        },
        "imaging": "imaging",
        "f": np.random.randint(-200, 200, (n_sessions, n_roi, len_session)),
        "is_cell": "is_cell",
        "r_neu": "r_neu",
        "stim": [
            {
                "screen_size": "screen_size",
                "stimulus": {
                    # ascii string
                    "grey_or_static": np.array(
                        [
                            103,
                            114,
                            101,
                            121,
                            95,
                            115,
                            116,
                            97,
                            116,
                            105,
                            99,
                            95,
                            100,
                            114,
                            105,
                            102,
                            116,
                        ]
                    ),
                    "n_baseline_triggers": n_baseline_triggers,
                    "directions": np.random.randint(
                        100, size=int(n_stim / n_sessions)
                    ),
                    "cycles_per_visual_degree": np.random.randint(
                        100, size=int(n_stim / n_sessions)
                    ),
                    "cycles_per_second": np.random.randint(
                        100, size=int(n_stim / n_sessions)
                    ),
                },
                "n_triggers": 2 * n_baseline_triggers
                + (n_stim / n_sessions) * n_triggers_per_stim,
            }
        ]
        * n_sessions,
        "trig": "trig",
    }
    data_raw = DataRaw(data, is_allen=False)

    yield data_raw


@pytest.fixture
def get_photon_data(get_data_raw_object, get_config):
    photon_data = PhotonData.__new__(PhotonData)
    photon_data.deactivate_checks = True
    photon_data.photon_type = PhotonType.TWO_PHOTON
    photon_data.config = get_config
    photon_data.fps = get_fps(photon_data.photon_type, photon_data.config)
    photon_data.set_general_variables(get_data_raw_object)

    yield photon_data


def test_set_general_variables(get_variables, get_photon_data):
    n_sessions, n_roi, len_session, n_stim, _, _ = get_variables
    photon_data = get_photon_data

    assert photon_data.n_sessions == n_sessions
    assert photon_data.n_roi == n_roi
    assert photon_data.n_frames_per_session == len_session
    assert photon_data.n_of_stimuli_per_session == n_stim / 2
    assert (
        photon_data.stimulus_start_frames.shape[0] == n_sessions * n_stim / 2
    )


def test_make_signal_dataframe(
    get_photon_data, get_data_raw_object, get_variables
):
    photon_data = get_photon_data
    signal = photon_data.make_signal_dataframe(get_data_raw_object)
    n_sessions, n_roi, len_session, _, _, _ = get_variables

    assert signal.shape == (len_session * n_sessions * n_roi, 10)


def test_get_stimuli(get_photon_data, get_data_raw_object, get_variables):
    _, _, _, n_stim, _, _ = get_variables
    photon_data = get_photon_data
    stimuli = photon_data.get_stimuli(get_data_raw_object)

    assert stimuli.shape == (n_stim, 4)


def test_fill_up_with_stim_info(get_photon_data, get_data_raw_object):
    photon_data = get_photon_data
    signal = photon_data.make_signal_dataframe(get_data_raw_object)
    stimuli = photon_data.get_stimuli(get_data_raw_object)
    signal = photon_data.fill_up_with_stim_info(signal, stimuli)

    frames = set(signal[signal["stimulus_onset"]].frames_id)

    assert frames == set(photon_data.stimulus_start_frames)
