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
    """
    This function returns the variables that are used to generate the mock
    data, and are designed to replicate the structure of the experiment.

    =======================================
    Description of the stimuli parameters
    =======================================
    In this simulated data we have blocks of 2 sessions (which can be repeated
    in multiple days), 2 spatial frequencies, 2 temporal frequencies and 2
    directions, for a total of 8 unique stimuli. Each session has 3 repetitions
    of each unique stimulus, for a total of 24 stimuli per session.

    As in the real data, the number of baseline triggers is 4, and the number
    of frames per trigger is 75.

    With this information, I made up a matrix of stimuli that is used to
    generate the mock data. The matrix is as follows:

    =======================================
    Matrix of sf / tf / direction stims
    =======================================
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

    Returns
    -------
    tuple(namedtuple, namedtuple, namedtuple)
        Nametuple used:
        - `Parameters`:
            - n_roi: number of rois
            - n_baseline_triggers: number of baseline triggers
            - n_triggers_per_stim: number of triggers per stimulus
            - n_frames_per_trigger: number of frames per trigger
            - n_unique_stim: number of unique stimuli
            - n_repetition: number of repetitions per unique stimulus
        - `SessionParameters`:
            - `sf`: spatial frequencies
            - `tf`: temporal frequencies
            - `dir`: directions
        Mores specifically the output is: (session_1, session_2, parameters)

        These named tuples contain parameters that are used by
        `make_stim_dict` to create the `stimulus` dict, which is a
        fundamental part of the mock data. The first two namedtuples contain
        the information regarding two unique sessions.
    """

    parameters = namedtuple(
        "Parameters",
        [
            "n_roi",
            "n_baseline_triggers",
            "n_triggers_per_stim",
            "n_frames_per_trigger",
            "n_unique_stim",
            "n_repetition",
        ],
    )

    p = parameters(
        n_roi=5,
        n_baseline_triggers=4,
        n_triggers_per_stim=3,
        n_frames_per_trigger=75,
        n_unique_stim=8,
        n_repetition=3,
    )

    sf_values = [1, 2]
    tf_values = [3, 4]
    dir_values = [5, 6]

    session = namedtuple("SessionParameters", ["sf", "tf", "dir"])

    session1 = session(
        sf=p.n_unique_stim * p.n_repetition * sf_values[0],
        tf=np.tile(
            [
                tf_values[0],
                tf_values[0],
                tf_values[1],
                tf_values[1],
            ],
            3,
        ),
        dir=np.tile(
            [
                dir_values[0],
                dir_values[1],
            ],
            6,
        ),
    )

    session2 = session(
        sf=p.n_unique_stim * p.n_repetition * sf_values[1],
        tf=session1.tf,
        dir=session1.dir,
    )

    return session1, session2, p


def make_variables_day_related(p, multiple_days=False):
    """
    This method returns other variables that are used to generate the mock
    data, that are influenced by the number of days of the experiment.
    The content of the session is not influenced by the number of days, but
    the number of sessions is, together with the number of stimuli.

    In order to better understand them, here a schematic of a session:

    =======================================
    Structure of a session
    =======================================
    --- Session 1 -----------------------------------
    === n_baseline_triggers =========================
    === a subset of n_unique_stim ===================
     repeated n_repetition times
     each stim contains n_triggers_per_stim triggers
     repeated for ech ROI
    === n_baseline_triggers =========================

    Parameters
    ----------
    p : namedtuple
        This is the namedtuple that contains the parameters of the experiment,
        as returned by `get_shared_variables_to_generate_mock_data`.
    multiple_days : bool, optional
        Whether the experiment has multiple days or not, by default False.
        If True, the `n_days` variable is 2, otherwise it is 1.

    Returns
    -------
    namedtuple
        This namedtuple contains the following variables:
        - n_sessions: number of sessions
        - len_session: length of a session
        - n_stim: number of stimuli
        - n_days: number of days
        - day_stimulus: stimulus of each day
    """

    if multiple_days:
        n_days = 2
    else:
        n_days = 1
    n_sessions = 2 * n_days
    n_stim = p.n_unique_stim * p.n_repetition * n_days

    len_session = int(
        (
            2 * p.n_baseline_triggers
            + n_stim / n_sessions * p.n_triggers_per_stim
        )
        * p.n_frames_per_trigger
    )

    day_stimulus = (
        [1] * p.n_roi + [2] * p.n_roi if multiple_days else [1] * p.n_roi
    )

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
    """
    This method returns the config file used to generate the mock data.
    It is a subset of the config file used in the real experiment.

    Returns
    -------
    dict
        Dictionary containing the config file.
    """
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
        "parser": "Parser2pRSP",
        "fitting": {
            "power_law_exp": 1,
            "lower_bounds": [-200, 0, 0, 0.01, 0.01, -np.inf],
            "upper_bounds": [np.inf, 20, 20, 4, 4, np.inf],
            "iterations_to_fit": 20,
            "jitter": 0.1,
            "oversampling_factor": 100,
        },
        "paths": {
            "imaging": "test_data/",
            "allen-dff": "test_data/allen_dff/",
            "serial2p": "test_data/serial2p/",
            "stimulus-ai-schedule": "test_data/stimulus_ai_schedule/",
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
    """
    This method returns the stimulus dictionary used to generate the mock data.
    It is based on the variables set by the methods
    `get_shared_variables_to_generate_mock_data` and
    `make_variables_day_related`.

    Parameters
    ----------
    n_baseline_triggers : int
        Number of baseline triggers.
    n_stim_per_session : int
        Number of stimuli per session.
    n_triggers_per_stim : int
        Number of triggers per stimulus.
    directions : list
        List of directions.
    cycles_per_visual_degree : list
        List of cycles per visual degree. Refers to the spatial frequency.
    cycles_per_second : list
        List of cycles per second. Refers to the temporal frequency.

    Returns
    -------
    dict
        Dictionary containing the stimulus information.
    """
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
    """
    This method generates random responses for the mock data starting from
    a seed number. The random numbers are drawn from a Poisson distribution
    with a random expected value. The sign of the responses is flipped
    randomly.

    Parameters
    ----------
    seed_number : int
        Seed number.
    n_sessions : int
        Number of sessions.
    n_roi : int
        Number of ROIs.
    len_session : int
        Length of a session.

    Returns
    -------
    np.ndarray
        Array containing the random responses. The shape is
        (n_sessions, n_roi, len_session). The information regarding the
        number of days is contained in the number of sessions. For different
        days different random responses are generated.
    """

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
    """
    This method assembles together the dicts and variables to generate the
    mock data that are produced thanks to the other methods in this module.
    It replicates the structure that `DataRaw` class is expected to receive.
    This allows us to have a mock data that is digestible by the same pipeline
    used for the real data.

    Parameters
    ----------
    seed_number : int
        Seed number used to generate the random responses.
    day_vars : namedtuple
        Namedtuple containing the variables related to the day, generated by
        the method `make_variables_day_related`.
    session1 : namedtuple
        Namedtuple containing the variables related to the first session,
        generated by the method `get_shared_variables_to_generate_mock_data`.
    session2 : namedtuple
        Namedtuple containing the variables related to the second session,
        generated by the method `get_shared_variables_to_generate_mock_data`.
    params : namedtuple
        Namedtuple containing the parameters used to generate the mock data,
        generated by the method `get_shared_variables_to_generate_mock_data`.

    Returns
    -------
    dict
        Dictionary containing the mock data.
    """
    return {
        "day": {
            "roi": "roi",
            "roi_label": "roi_label",
            "stimulus": day_vars.day_stimulus,
        },
        "imaging": "imaging",
        "f": make_random_responses(
            seed_number,
            day_vars.n_sessions,
            params.n_roi,
            day_vars.len_session,
        ),
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
    """
    This method generates a mock `DataRaw` object with simulated data.

    Parameters
    ----------
    seed_number : int, optional
        The seed number used to generate the random responses, by default 1
    multiple_days : bool, optional
        Weather to generate data for multiple days or not, by default False

    Returns
    -------
    DataRaw
        The mock `DataRaw` object.
    """
    session1, session2, params = get_shared_variables_to_generate_mock_data()
    day_relate_vars = make_variables_day_related(params, multiple_days)

    data = make_raw_data_dict_mock(
        seed_number,
        day_relate_vars,
        session1,
        session2,
        params,
    )
    data_raw = DataRaw(data, is_summary_data=False)

    return data_raw


def get_photon_data_mock(seed_number=1, multiple_days=False):
    """
    This method generates a mock `PhotonData` object with simulated data.

    Parameters
    ----------
    seed_number : int, optional
        The seed number used to generate the random responses, by default 1
    multiple_days : bool, optional
        Weather to generate data for multiple days or not, by default False

    Returns
    -------
    PhotonData
        The mock `PhotonData` object.
    """
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
    """
    This method generates a mock `FrequencyResponsiveness` object with
    simulated data.

    Parameters
    ----------
    seed_number : int, optional
        The seed number used to generate the random responses, by default 1
    multiple_days : bool, optional
        Weather to generate data for multiple days or not, by default False

    Returns
    -------
    FrequencyResponsiveness
        The mock `FrequencyResponsiveness` object.
    """
    pt = PhotonType.TWO_PHOTON
    return FrequencyResponsiveness(
        PhotonData(
            data_raw=get_data_raw_object_mock(seed_number, multiple_days),
            photon_type=pt,
            config=get_config_mock(),
            using_real_data=False,
        )
    )
