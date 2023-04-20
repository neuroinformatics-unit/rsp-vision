import numpy as np

from rsp_vision.analysis.spatial_freq_temporal_freq import (
    FrequencyResponsiveness,
)
from rsp_vision.load.config_switches import get_fps
from rsp_vision.objects.data_raw import DataRaw
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData


def get_experimental_variables_mock():
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


def get_config_mock():
    return {
        "fps_two_photon": 30,
        "trigger_interval_s": 2.5,
        "n_spatial_frequencies": 6,
        "n_temporal_frequencies": 6,
        "n_directions": 8,
        "padding": [0, 1],
        "baseline": "static",
        "anova_threshold": 0.1,
        "response_magnitude_threshold": 0.1,
        "consider_only_positive": False,
        "only_positive_threshold": 0.1,
        "spatial_frequencies": [0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
        "temporal_frequencies": [0.5, 1, 2, 4, 8, 16],
        "directions": [0, 45, 90, 135, 180, 225, 270, 315],
        "fitting": {
            "power_law_exp": 1,
            "lower_bounds": [-200, 0, 0, 0.01, 0.01, -np.inf],
            "upper_bounds": [np.inf, 20, 20, 4, 4, np.inf],
            "iterations_to_fit": 20,
            "jitter": 0.1,
            "oversampling_factor": 100,
        },
    }


def get_stim_mock(
    n_baseline_triggers,
    n_stim_per_session,
    n_triggers_per_stim,
    directions,
    cycles_per_visual_degree,
    cycles_per_second,
):
    return {
        "screen_size": "screen_size",
        "stimulus": {
            "grey_or_static": np.frombuffer(
                b"grey_static_drift", dtype=np.uint8
            ),
            "n_baseline_triggers": n_baseline_triggers,
            "directions": np.array(directions),
            "cycles_per_visual_degree": np.array(cycles_per_visual_degree),
            "cycles_per_second": np.array(cycles_per_second),
        },
        "n_triggers": 2 * n_baseline_triggers
        + n_stim_per_session * n_triggers_per_stim,
    }


def get_random_responses(seed_number, n_sessions, n_roi, len_session):
    np.random.seed(seed_number)
    return np.random.randint(-200, 200, (n_sessions, n_roi, len_session))


def get_raw_data_dict_mock(
    n_sessions,
    n_roi,
    len_session,
    n_stim,
    n_baseline_triggers,
    n_triggers_per_stim,
    seed_number,
):
    return {
        "day": {
            "roi": "roi",
            "roi_label": "roi_label",
            "stimulus": "stimulus",
        },
        "imaging": "imaging",
        "f": get_random_responses(seed_number, n_sessions, n_roi, len_session),
        "is_cell": "is_cell",
        "r_neu": "r_neu",
        "stim": [
            get_stim_mock(
                n_baseline_triggers,
                n_stim // n_sessions,
                n_triggers_per_stim,
                np.ones(12),  # half stim combinations in this session
                np.tile([3, 3, 4, 4], 3),
                np.tile([6, 5], 6),
            ),
            get_stim_mock(
                n_baseline_triggers,
                n_stim // n_sessions,
                n_triggers_per_stim,
                np.ones(12) * 2,
                np.tile([3, 3, 4, 4], 3),
                np.tile([6, 5], 6),
            ),
        ],
        "trig": "trig",
    }


def get_data_raw_object_mock(seed_number=1):
    (
        n_sessions,
        n_roi,
        len_session,
        n_stim,
        n_baseline_triggers,
        n_triggers_per_stim,
    ) = get_experimental_variables_mock()

    data = get_raw_data_dict_mock(
        n_sessions,
        n_roi,
        len_session,
        n_stim,
        n_baseline_triggers,
        n_triggers_per_stim,
        seed_number,
    )
    data_raw = DataRaw(data, is_allen=False)

    return data_raw


# not sure I need this one
def get_photon_data_mock(seed_number=1):
    photon_data = PhotonData.__new__(PhotonData)
    photon_data.using_real_data = False
    photon_data.photon_type = PhotonType.TWO_PHOTON
    photon_data.config = get_config_mock()
    photon_data.fps = get_fps(photon_data.photon_type, photon_data.config)
    data_raw = get_data_raw_object_mock(seed_number)
    photon_data.set_general_variables(data_raw)

    return photon_data


def get_response_mock(seed_number):
    pt = PhotonType.TWO_PHOTON
    return FrequencyResponsiveness(
        PhotonData(
            data_raw=get_data_raw_object_mock(seed_number),
            photon_type=pt,
            config=get_config_mock(),
            using_real_data=False,
        )
    )
