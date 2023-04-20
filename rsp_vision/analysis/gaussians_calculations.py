from math import exp, log2

import numpy as np

from rsp_vision.dashboard.query_dataframes import get_preferred_sf_tf


def fit_elliptical_gaussian(sfs_inverted, tfs, responses, roi_id, config, dir):
    sf_0, tf_0, peak_response = get_preferred_sf_tf(responses, roi_id, dir)

    # same tuning width for sf and tf
    sigma = config["fitting"]["tuning_width"]

    R = np.zeros((len(sfs_inverted), len(tfs)))
    for i, sf in enumerate(sfs_inverted):
        for j, tf in enumerate(tfs):
            R[i, j] = elliptical_gaussian_adermann(
                peak_response, sf, tf, sf_0, tf_0, sigma
            )

    return R


def elliptical_gaussian_adermann(peak_response, sf, tf, sf_0, tf_0, sigma):
    r = (
        peak_response
        * exp(-((log2(sf) - log2(sf_0)) ** 2) / 2 * (sigma**2))
        * exp(-((log2(tf) - log2(tf_0)) ** 2) / 2 * (sigma**2))
    )

    return r
