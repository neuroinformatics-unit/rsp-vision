import numpy as np
import pytest

from rsp_vision.analysis.spatial_freq_temporal_freq import (
    FrequencyResponsiveness,
)
from rsp_vision.load.config_switches import get_fps
from rsp_vision.objects.data_raw import DataRaw
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData


@pytest.fixture
def get_variables():
    n_sessions = 2  # if you change this please adapt "stim" accordingly
    n_roi = 5
    n_stim = 24  # 8 stim and 3 rep for all
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
        "padding": [0, 1],
        "baseline": "static",
        "anova_threshold": 0.1,
        "response_magnitude_threshold": 0.1,
        "consider_only_positive": True,
        "only_positive_threshold": 0.1,
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

    np.random.seed(101)

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
        #  number of stim == n_session
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
                    "directions": np.array(
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    ),
                    "cycles_per_visual_degree": np.array(
                        [3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4]
                    ),
                    "cycles_per_second": np.array(
                        [6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5]
                    ),
                },
                "n_triggers": 2 * n_baseline_triggers
                + (n_stim / n_sessions) * n_triggers_per_stim,
            },
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
                    "directions": np.array(
                        [
                            2,
                            2,
                            2,
                            2,
                            2,
                            2,
                            2,
                            2,
                            2,
                            2,
                            2,
                            2,
                        ]
                    ),
                    "cycles_per_visual_degree": np.array(
                        [3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4]
                    ),
                    "cycles_per_second": np.array(
                        [6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5]
                    ),
                },
                "n_triggers": 2 * n_baseline_triggers
                + (n_stim / n_sessions) * n_triggers_per_stim,
            },
        ],
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


@pytest.fixture
def get_freq_response_instance(get_config, get_data_raw_object):
    pt = PhotonType.TWO_PHOTON
    yield FrequencyResponsiveness(
        PhotonData(
            data_raw=get_data_raw_object,
            photon_type=pt,
            config=get_config,
            deactivate_checks=True,
        )
    )
