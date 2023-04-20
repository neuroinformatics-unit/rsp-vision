from math import exp, log2

import numpy as np
from scipy.optimize import least_squares


def symmetric_2D_gaussian(peak_response, sf, tf, sf_0, tf_0, sigma):
    r = (
        peak_response
        * exp(-((log2(sf) - log2(sf_0)) ** 2) / 2 * (sigma**2))
        * exp(-((log2(tf) - log2(tf_0)) ** 2) / 2 * (sigma**2))
    )

    return r


def elliptical_gaussian_andermann(
    peak_response, sf, tf, sf_0, tf_0, sigma_sf, sigma_tf, 𝜻_power_law_exp
):
    # might want to normalize the Log2

    log_2_tf_pref_sf = 𝜻_power_law_exp * (log2(sf) - log2(sf_0)) + log2(tf_0)

    r = (
        peak_response
        * exp(-((log2(sf) - log2(sf_0)) ** 2) / 2 * (sigma_sf**2))
        * exp(-((log2(tf) - log_2_tf_pref_sf) ** 2) / 2 * (sigma_tf**2))
    )

    return r


def fit_2D_gaussian_to_data(
    sfs_inverted, tfs, response_matrix, parameters_to_fit
):
    (
        peak_response,
        sf_0,
        tf_0,
        sigma_sf,
        sigma_tf,
        𝜻_power_law_exp,
    ) = parameters_to_fit
    jitter = peak_response / 2

    results = []
    for _ in range(20):
        params_for_andermann = np.array(
            [
                peak_response,
                sfs_inverted,
                tfs,
                sf_0,
                tf_0,
                sigma_sf,
                sigma_tf,
                𝜻_power_law_exp,
            ]
        )
        results.append(
            least_squares(
                elliptical_gaussian_andermann,
                params_for_andermann,
                method="lm",
            )
        )
        peak_response, sf_0, tf_0, sigma_sf, sigma_tf, 𝜻_power_law_exp = [
            peak_response,
            sf_0,
            tf_0,
            sigma_sf,
            sigma_tf,
            𝜻_power_law_exp,
        ] + jitter * np.random.rand(1)
