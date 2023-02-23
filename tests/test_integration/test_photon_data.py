import numpy as np
import pytest

from rsp_vision.objects.data_raw import DataRaw
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData


@pytest.fixture
def get_variables():
    n_sessions = 18
    n_roi = 11
    l_signal = 11400
    n_stim = 48

    return n_sessions, n_roi, l_signal, n_stim


@pytest.fixture
def get_config():
    yield {"fps_two_photon": 30}


@pytest.fixture
def get_data_raw_object(get_variables):
    n_sessions, n_roi, l_signal, n_stim = get_variables

    data = {
        "day": {
            "roi": "roi",
            "roi_label": "roi_label",
            "stimulus": "stimulus",
        },
        "imaging": "imaging",
        # it should be some kind of range
        "f": np.ones((n_sessions, n_roi, l_signal)),
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
                    "n_baseline_triggers": 4,
                    "directions": np.arange(0, n_stim),
                    "cycles_per_visual_degree": np.arange(0, n_stim),
                    "cycles_per_second": np.arange(0, n_stim),
                },
                "n_triggers": 152,
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
    photon_data.photon_type = PhotonType.TWO_PHOTON
    photon_data.config = get_config
    photon_data.set_general_variables(get_data_raw_object)

    yield photon_data


def test_set_general_variables(get_variables, get_photon_data):
    n_sessions, n_roi, l_signal, n_stim = get_variables
    photon_data = get_photon_data

    assert photon_data.n_sessions == n_sessions
    assert photon_data.n_roi == n_roi
    assert photon_data.n_frames_per_session == l_signal
    assert photon_data.n_of_stimuli_per_session == n_stim
    assert photon_data.stimulus_start_frames.shape[0] == n_sessions * n_stim


def test_make_signal_dataframe(get_photon_data, get_data_raw_object):
    photon_data = get_photon_data
    signal = photon_data.make_signal_dataframe(get_data_raw_object)

    assert signal.shape == (2257200, 10)


def test_get_stimuli(get_photon_data, get_data_raw_object):
    photon_data = get_photon_data
    stimuli = photon_data.get_stimuli(get_data_raw_object)

    assert stimuli.shape == (864, 4)


def test_fill_up_with_stim_info(get_photon_data, get_data_raw_object):
    photon_data = get_photon_data
    signal = photon_data.make_signal_dataframe(get_data_raw_object)
    stimuli = photon_data.get_stimuli(get_data_raw_object)
    signal = photon_data.fill_up_with_stim_info(signal, stimuli)

    subset = signal[signal["stimulus_onset"]].iloc[7]

    assert subset["sf"] == 7
    assert subset["tf"] == 7
    assert subset["direction"] == 7
