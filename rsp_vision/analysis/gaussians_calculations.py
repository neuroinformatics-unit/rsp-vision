from math import exp, log2


def elliptical_gaussian_adermann(peak_response, sf, tf, sf_0, tf_0, sigma):
    r = (
        peak_response
        * exp(-((log2(sf) - log2(sf_0)) ** 2) / 2 * (sigma**2))
        * exp(-((log2(tf) - log2(tf_0)) ** 2) / 2 * (sigma**2))
    )

    return r
