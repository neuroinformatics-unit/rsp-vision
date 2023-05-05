import numpy as np
from numba import njit
from scipy.optimize import OptimizeResult, least_squares


@njit
def elliptical_gaussian_andermann(
    peak_response: float,
    sf: float,
    tf: float,
    sf_0: float,
    tf_0: float,
    sigma_sf: float,
    sigma_tf: float,
    𝜻_power_law_exp: float,
) -> float:
    """Calculate the response of a neuron to a visual stimulus using a two-
    dimensional elliptical Gaussian function, as described in Andermann et
    al., 2011.

    Parameters
    ----------
    peak_response : float
        Peak response of the neuron.
    sf : float
        Spatial frequency of the stimulus.
    tf : float
        Temporal frequency of the stimulus.
    sf_0 : float
        Preferred spatial frequency of the neuron.
    tf_0 : float
        Preferred temporal frequency of the neuron.
    sigma_sf : float
        Spatial frequency tuning width of the neuron.
    sigma_tf : float
        Temporal frequency tuning width of the neuron.
    𝜻_power_law_exp : float
        Exponent for the power law that governs the temporal frequency
        preference of the neuron with respect to spatial frequency.

    Returns
    -------
    float
        Response of the neuron to the stimulus.
    """
    log_2_tf_pref_sf = 𝜻_power_law_exp * (
        np.log2(sf) - np.log2(sf_0)
    ) + np.log2(tf_0)

    r = (
        peak_response
        * np.exp(-((np.log2(sf) - np.log2(sf_0)) ** 2) / 2 * (sigma_sf**2))
        * np.exp(
            -((np.log2(tf) - log_2_tf_pref_sf) ** 2) / 2 * (sigma_tf**2)
        )
    )

    return r


def single_fit(
    params: np.ndarray,
    sfs: np.ndarray,
    tfs: np.ndarray,
    response_matrix: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> OptimizeResult:
    """Fit a two-dimensional Gaussian model to a response matrix using
    least squares optimization.

    Parameters
    ----------
    params : np.ndarray
        An array containing the initial values of the Gaussian model
        parameters to be optimized.
    sfs : np.ndarray
        An array containing the spatial frequencies of the stimuli.
    tfs : np.ndarray
        An array containing the temporal frequencies of the stimuli.
    response_matrix : np.ndarray
        A matrix of recorded responses to the stimuli.
    lower_bounds : np.ndarray
        An array of lower bounds for the optimized parameters.
    upper_bounds : np.ndarray
        An array of upper bounds for the optimized parameters.

    Returns
    -------
    OptimizeResult
        A scipy.optimize.OptimizeResult object containing information about
        the optimization procedure and results.
    """

    def cost_function(params):
        model = create_gaussian_matrix(params, sfs, tfs)
        residuals = response_matrix - model
        return residuals.ravel()

    result = least_squares(
        cost_function,
        params,
        method="trf",
        bounds=(lower_bounds, upper_bounds),
    )
    return result


def fit_2D_gaussian_to_data(
    sfs: np.ndarray,
    tfs: np.ndarray,
    response_matrix: np.ndarray,
    parameters_to_fit: np.ndarray,
    config: dict,
) -> OptimizeResult:
    """Fit a two-dimensional Gaussian model to a matrix of recorded
    responses to visual stimuli.

    Parameters
    ----------
    sfs : np.ndarray
        An array of spatial frequencies of the stimuli.
    tfs : np.ndarray
        An array of temporal frequencies of the stimuli.
    response_matrix : np.ndarray
        A matrix of recorded responses to the stimuli.
    parameters_to_fit : np.ndarray
        An array of initial values for the parameters of the Gaussian model to
        be optimized.
    config : dict
        A dictionary containing the configuration parameters for the fitting
        procedure. The configuration parameters are:
        - "fitting": a dictionary containing the following keys:
            - "jitter": a float representing the scaling factor for perturbing
            the initial parameters
            - "iterations_to_fit": an integer representing the number of times
            to run the optimization procedure
            - "lower_bounds": an array of lower bounds for the optimized
            parameters
            - "upper_bounds": an array of upper bounds for the optimized
            parameters

    Returns
    -------
    OptimizeResult
        A Scipy OptimizeResult object containing information about the
        optimization procedure and results.
    """
    jitter = np.array([parameters_to_fit]) * config["fitting"]["jitter"]
    best_result = None
    best_residuals = float("inf")

    for _ in range(config["fitting"]["iterations_to_fit"]):
        # Add jitter to the initial parameters
        perturbation = np.random.randn(1) * jitter
        perturbed_params = np.maximum(
            0, parameters_to_fit + perturbation
        ).tolist()[0]

        # Ensure sf_0 and tf_0 remain positive
        perturbed_params[1] = max(perturbed_params[1], 1e-5)
        perturbed_params[2] = max(perturbed_params[2], 1e-5)
        lower_bounds = config["fitting"]["lower_bounds"]
        upper_bounds = config["fitting"]["upper_bounds"]
        perturbed_params = np.clip(
            perturbed_params, lower_bounds, upper_bounds
        )

        # Fit the model with the perturbed parameters
        result = single_fit(
            perturbed_params,
            sfs,
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


def create_gaussian_matrix(
    params: np.ndarray,
    sfs: np.ndarray,
    tfs: np.ndarray,
) -> np.ndarray:
    """Create a matrix of Gaussian response amplitudes to a set of visual
    stimuli.

    Parameters
    ----------
    params : np.ndarray
        An array of parameters for the elliptical Gaussian model.
        The array must contain the following elements in the specified
        order:
        - peak_response: float, the peak response amplitude
        - sf_0: float, the preferred spatial frequency
        - tf_0: float, the preferred temporal frequency
        - sigma_sf: float, the spatial frequency tuning width
        - sigma_tf: float, the temporal frequency tuning width
        - 𝜻_power_law_exp: float, the exponent controlling the dependence of
        temporal frequency preference on spatial frequency
    sfs : np.ndarray
        An array of spatial frequencies of the stimuli.
    tfs : np.ndarray
        An array of temporal frequencies of the stimuli.

    Returns
    -------
    np.ndarray
        A 2D numpy array of Gaussian response amplitudes to the set of visual
        stimuli.
    """
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
