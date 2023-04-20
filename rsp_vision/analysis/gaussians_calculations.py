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
    log_2_tf_pref_sf = 𝜻_power_law_exp * (log2(sf) - log2(sf_0)) + log2(tf_0)

    r = (
        peak_response
        * exp(-((log2(sf) - log2(sf_0)) ** 2) / 2 * (sigma_sf**2))
        * exp(-((log2(tf) - log_2_tf_pref_sf) ** 2) / 2 * (sigma_tf**2))
    )

    return r


def single_fit(
    params, sfs_inverted, tfs, response_matrix, lower_bounds, upper_bounds
):
    # Define the cost function
    def cost_function(params):
        model = create_gaussian_matrix(params, sfs_inverted, tfs)
        residuals = response_matrix - model
        return residuals.ravel()

    # Fit the model using least_squares
    result = least_squares(
        cost_function,
        params,
        method="trf",
        bounds=(lower_bounds, upper_bounds),
    )
    return result


def fit_2D_gaussian_to_data(
    sfs_inverted, tfs, response_matrix, parameters_to_fit
):
    jitter = np.array([parameters_to_fit]) * 0.1
    best_result = None
    best_residuals = float("inf")

    # Loop 20 times and find the best fit
    for _ in range(20):
        # Add jitter to the initial parameters
        # perturbed_params = parameters_to_fit + jitter * np.random.rand(1)
        perturbation = np.random.randn(1) * jitter
        perturbed_params = np.maximum(
            0, parameters_to_fit + perturbation
        ).tolist()[0]

        # Ensure sf_0 and tf_0 remain positive
        perturbed_params[1] = max(perturbed_params[1], 1e-5)
        perturbed_params[2] = max(perturbed_params[2], 1e-5)
        lower_bounds = [-200, 0, 0, 0.01, 0.01, -np.inf]
        upper_bounds = [np.inf, 20, 20, 4, 4, np.inf]
        perturbed_params = np.clip(
            perturbed_params, lower_bounds, upper_bounds
        )

        # Fit the model with the perturbed parameters
        result = single_fit(
            perturbed_params,
            sfs_inverted,
            tfs,
            response_matrix,
            lower_bounds,
            upper_bounds,
        )

        # Keep track of the best result
        if result.success and result.cost < best_residuals:
            best_residuals = result.cost
            best_result = result

    # Return the best result
    return best_result


def create_gaussian_matrix(params, sfs, tfs):
    peak_response, sf_0, tf_0, sigma_sf, sigma_tf, 𝜻_power_law_exp = params

    gaussian_matrix = np.zeros((len(sfs), len(tfs)))

    for i, sf in enumerate(sfs):
        for j, tf in enumerate(tfs):
            gaussian_matrix[i, j] = elliptical_gaussian_andermann(
                peak_response,
                sf,
                tf,
                sf_0,
                tf_0,
                sigma_sf,
                sigma_tf,
                𝜻_power_law_exp,
            )

    return gaussian_matrix
