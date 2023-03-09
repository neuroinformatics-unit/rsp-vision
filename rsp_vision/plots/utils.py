from math import exp, log2

import numpy as np


def get_preferred_sf_tf(responses, roi_id, dir):
    median_subtracted_response = (
        responses[(responses.roi_id == roi_id) & (responses.direction == dir)]
        .groupby(["sf", "tf"])[["subtracted"]]
        .median()
    )
    sf_0, tf_0 = median_subtracted_response["subtracted"].idxmax()
    peak_response = median_subtracted_response.loc[(sf_0, tf_0)]["subtracted"]
    return sf_0, tf_0, peak_response


def fit_elliptical_gaussian(uniques, responses, roi_id, config, dir):
    sfs = np.sort(uniques["sf"])[::-1]
    tfs = np.sort(uniques["tf"])
    sf_0, tf_0, peak_response = get_preferred_sf_tf(responses, roi_id, dir)

    # same tuning width for sf and tf
    sigma = config["fitting"]["tuning_width"]

    R = np.zeros((len(sfs), len(tfs)))
    for i, sf in enumerate(sfs):
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
