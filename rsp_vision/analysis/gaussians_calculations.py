import logging
import sys

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


def get_gaussian_matrix_to_be_plotted(
    kind: str,
    roi_id: int,
    fit_output: dict,
    sfs: np.ndarray,
    tfs: np.ndarray,
    pooled_directions: bool = False,
    direction: float = sys.float_info.max,
    matrix_definition: int = 100,
) -> np.ndarray:
    """Returns a squared Gaussian matrix to be visualized in the dashboard
    based on the fitting parameters precalculated.
    To get a matrix for a single direction, set pooled_directions to False and
    specify the direction. To get a matrix for all directions, set
    pooled_directions to True.
    `Kind` can be either "6x6 matrix" or "custom". If "6x6 matrix", the
    Gaussian matrix will be calculated for the expetimental values of spatial
    and temporal frequencies. If "custom", the Gaussian matrix will be
    calculated for the values of spatial and temporal frequencies calculated
    with `np.linspace(0, 1, matrix_definition)`. You can specify the
    dimension of the squared matrix with the `matrix_definition` parameter.

    Parameters
    ----------
    kind : str
        Either "6x6 matrix" or "custom".
    roi_id : int
        Which ROI to calculate the Gaussian matrix for.
    fit_output : dict
        A dictionary containing the fitting parameters for each ROI.
    sfs : np.ndarray
        Sorted array of spatial frequencies.
    tfs : np.ndarray
        Sorted array of temporal frequencies.
    pooled_directions : bool, optional
        Whether to pool the directions or not, by default False
    direction : float, optional
        If pooled_directions is False, the direction to calculate the Gaussian
        matrix for, by default sys.float_info.max
    matrix_definition : int, optional
        The dimension of the squared matrix, by default 100. Used if
        `kind="custom"`.

    Returns
    -------
    np.ndarray
        A squared Gaussian matrix to be visualized in the dashboard.

    Raises
    ------
    ValueError
        If kind is not "6x6 matrix" or "custom".
    """
    if kind == "6x6 matrix":
        if not pooled_directions:
            assert (
                direction != sys.float_info.max
            ), "direction must be specified"
            matrix = create_gaussian_matrix(
                fit_output[(roi_id, direction)],
                sfs,
                tfs,
            )
        else:
            matrix = create_gaussian_matrix(
                fit_output[(roi_id, "pooled")],
                sfs,
                tfs,
            )
    elif kind == "custom":
        logging.info(
            "Creating custom matrix with definition %d", matrix_definition
        )
        if not pooled_directions:
            assert (
                direction != sys.float_info.max
            ), "direction must be specified"
            matrix = create_gaussian_matrix(
                fit_output[(roi_id, direction)],
                np.linspace(
                    sfs.min(),
                    tfs.max(),
                    num=matrix_definition,
                ),
                np.linspace(
                    tfs.min(),
                    tfs.max(),
                    num=matrix_definition,
                ),
            )
        else:
            matrix = create_gaussian_matrix(
                fit_output[(roi_id, "pooled")],
                np.linspace(
                    sfs.min(),
                    tfs.max(),
                    num=matrix_definition,
                ),
                np.linspace(
                    tfs.min(),
                    tfs.max(),
                    num=matrix_definition,
                ),
            )
    else:
        raise ValueError("kind must be '6x6 matrix' or 'custom'")
    return matrix
