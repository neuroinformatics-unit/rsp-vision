from collections import namedtuple

import numpy as np

from rsp_vision.analysis.spatial_freq_temporal_freq import (
    FrequencyResponsiveness,
)
from rsp_vision.load.config_switches import get_fps
from rsp_vision.objects.data_raw import DataRaw
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData



def get_shared_variables_to_generate_mock_data():
    """_summary_

    Matrix of sf / tf / direction stims
    Day 1 and day 2 are identical
    Session 1 (combinations are repeated 3 times)
    sf:   1 1 1  1 1 1  1 1 1  1 1 1
    tf:   3 3 4  4 3 3  4 4 3  3 4 4
    dir:  6 5 6  5 6 5  6 5 6  5 6 5
          *        *        *
    Session 2 (combinations are repeated 3 times)
    sf:   2 2 2  2 2 2  2 2 2  2 2 2
    tf:   3 3 4  4 3 3  4 4 3  3 4 4
    dir:  6 5 6  5 6 5  6 5 6  5 6 5

    All these variables are used locally to generate the mock data
    and are retreived from there by the tests
    It is useful to have them here more ecplicit for debugging purposes

    Returns
    -------
    _type_
        _description_
    """
    

    parameters = namedtuple( "Parameters", [
        "n_roi", 
        "n_baseline_triggers", 
        "n_triggers_per_stim", 
        "n_frames_per_trigger",
        "n_unique_stim",
        "n_repetition",
        ])

    p = parameters(
        n_roi = 5,
        n_baseline_triggers = 4,
        n_triggers_per_stim = 3,
        n_frames_per_trigger = 75,
        n_unique_stim = 8,
        n_repetition = 3,
    )

    sf_values = [1, 2]
    tf_values = [3, 4]
    dir_values = [5, 6]

    session = namedtuple( "SessionParameters", [
        "sf", "tf", "dir"
        ])
    
    session1 = session(
        sf =  p.n_unique_stim * p.n_repetition * sf_values[0],
        tf =  np.tile(
            [
                tf_values[0],
                tf_values[0],
                tf_values[1],
                tf_values[1],
            ], 3),
        dir = np.tile(
            [
                dir_values[0],
                dir_values[1],
            ],6,),
    )

    session2 = session(
        sf =  p.n_unique_stim * p.n_repetition * sf_values[1],
        tf =  session1.tf,
        dir = session1.dir,
    )

    return session1, session2, p
        

def make_variables_day_related(p, multiple_days=False):
    """
    Structure of a session
    --- Session 1 -----------------------------------
    === n_baseline_triggers =========================
    === a subset of n_unique_stim ===================
     repeated n_repetition times
     each stim contains n_triggers_per_stim triggers
     repeated for ech ROI
    === n_baseline_triggers =========================

    Parameters
    ----------
    p : _type_
        _description_
    multiple_days : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    
    
    if multiple_days:
        n_days = 2
    else:
        n_days = 1
    n_sessions = 2 * n_days
    n_stim = p.n_unique_stim * p.n_repetition * n_days

    len_session = int(
        (2 * p.n_baseline_triggers + n_stim / n_sessions * p.n_triggers_per_stim)
        * p.n_frames_per_trigger
    )

    day_stimulus = [1] * p.n_roi + [2] * p.n_roi if multiple_days else [1] * p.n_roi

    variables = namedtuple(
        "Variables",
        [
            "n_sessions",
            "len_session",
            "n_stim",
            "n_days",
            "day_stimulus",
        ],
    )

    return variables(
        n_sessions=n_sessions,
        len_session=len_session,
        n_stim=n_stim,
        n_days=n_days,
        day_stimulus=day_stimulus,
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
        "fitting": {
            "power_law_exp": 1,
            "lower_bounds": [-200, 0, 0, 0.01, 0.01, -np.inf],
            "upper_bounds": [np.inf, 20, 20, 4, 4, np.inf],
            "iterations_to_fit": 20,
            "jitter": 0.1,
            "oversampling_factor": 100,
        },
    }


def make_stim_dict(
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


def make_random_responses(seed_number, n_sessions, n_roi, len_session):
    np.random.seed(seed_number)
    expected_value = np.abs(np.random.randint(50))  # use positive values only
    data = np.random.poisson(
        lam=expected_value, size=(n_sessions, n_roi, len_session)
    )
    data *= np.random.choice([-1, 1], size=data.shape)  # flip sign randomly
    return data


def make_raw_data_dict_mock(
    seed_number,
    day_vars,
    session1,
    session2,
    params,
):
    return {
        "day": {
            "roi": "roi",
            "roi_label": "roi_label",
            "stimulus": day_vars.day_stimulus,
        },
        "imaging": "imaging",
        "f": make_random_responses(seed_number, day_vars.n_sessions, params.n_roi, params.len_session),
        "is_cell": "is_cell",
        "r_neu": "r_neu",
        "stim": [
            make_stim_dict(
                params.n_baseline_triggers,
                day_vars.n_stim // day_vars.n_sessions,
                params.n_triggers_per_stim,
                session1.sf,
                session1.tf,
                session1.dir,
            ),
            make_stim_dict(
                params.n_baseline_triggers,
                day_vars.n_stim // day_vars.n_sessions,
                params.n_triggers_per_stim,
                session2.sf,
                session2.tf,
                session2.dir,
            ),
        ]
        * day_vars.n_days,
        "trig": "trig",
    }


def get_data_raw_object_mock(seed_number=1, multiple_days=False):
    session1, session2, params = get_shared_variables_to_generate_mock_data(multiple_days)
    day_relate_vars = make_variables_day_related(params, multiple_days)
    
    data = make_raw_data_dict_mock(
        seed_number,
        day_relate_vars,
        session1,
        session2,
        params,
    )
    data_raw = DataRaw(data, is_allen=False)

    return data_raw


def get_photon_data_mock(seed_number=1, multiple_days=False):
    photon_data = PhotonData.__new__(PhotonData)
    photon_data.using_real_data = False
    photon_data.photon_type = PhotonType.TWO_PHOTON
    photon_data.config = get_config_mock()
    photon_data.fps = get_fps(photon_data.photon_type, photon_data.config)
    data_raw = get_data_raw_object_mock(
        seed_number, multiple_days=multiple_days
    )
    photon_data.set_general_variables(data_raw)

    return photon_data


def get_response_mock(seed_number, multiple_days=False):
    pt = PhotonType.TWO_PHOTON
    return FrequencyResponsiveness(
        PhotonData(
            data_raw=get_data_raw_object_mock(seed_number, multiple_days),
            photon_type=pt,
            config=get_config_mock(),
            using_real_data=False,
        )
    )
