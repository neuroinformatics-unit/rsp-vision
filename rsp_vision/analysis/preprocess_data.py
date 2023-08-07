import logging

import allensdk.brain_observatory.dff as dff_module
import numpy as np
from allensdk.brain_observatory.r_neuropil import NeuropilSubtract

from rsp_vision.load.load_raw_suite2p_data import read_numpy_output_of_suite2p
from rsp_vision.load.load_stimulus_info import how_many_days_in_dataset


def pipeline_for_processing_raw_suite2p_data(folder_naming, config):
    n_days = how_many_days_in_dataset(folder_naming)
    data = read_numpy_output_of_suite2p(folder_naming, n_days)
    n_roi = len(data[1]["F"])

    # now handling only the case of one day. Need to extend to multiple
    # days using registers2p

    dff, r = neuropil_subtraction(
        data[1]["F"],
        data[1]["Fneu"],
    )

    len_session = config["len-session"]
    n_sessions = config["n-sessions"]
    dff = dff[:, :, np.newaxis]
    dff = dff.reshape(n_sessions, n_roi, len_session)

    is_cell = data[1]["iscell"]
    day = {"stimulus": [1] * n_roi}

    return {
        "f": dff,
        "is_cell": is_cell,
        "day": day,
        "r_neu": r,
        "imaging": ["suite2p"],
        "stim": ["TODO"],
        "trig": ["TODO"],
    }


def neuropil_subtraction(f, f_neu):
    # if fluorescence type is allen_df
    logging.info("Performing neuropil subtraction...")

    #  use default parameters for all methods
    neuropil_subtraction = NeuropilSubtract()
    neuropil_subtraction.set_F(f, f_neu)
    neuropil_subtraction.fit()

    r = neuropil_subtraction.r

    f_corr = f - r * f_neu

    # kernel values to be changed for 3-photon data
    # median_kernel_long = 1213, median_kernel_short = 23

    dff = 100 * dff_module.compute_dff_windowed_median(f_corr)

    return dff, r
