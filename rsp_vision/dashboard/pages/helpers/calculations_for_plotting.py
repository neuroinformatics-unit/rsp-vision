import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rsp_vision.analysis.gaussians_calculations import (
    get_gaussian_matrix_to_be_plotted,
)


def fit_correlation(
    gaussian: np.ndarray, median_subtracted_response: np.ndarray
) -> float:
    """This method calculates the Pearsons correlation between the median
    subtracted response and the Gaussian fit (6x6 matrix).

    Parameters
    ----------
    gaussian : np.ndarray
        The Gaussian fit (6x6 matrix).
    median_subtracted_response : np.ndarray
        The median subtracted response.

    Returns
    -------
    float
        The correlation between the median subtracted response and the Gaussian
        fit (6x6 matrix).
    """
    fit_corr, _ = pearsonr(
        median_subtracted_response.flatten(), gaussian.flatten()
    )
    return fit_corr


def calculate_mean_and_median(
    signal: pd.DataFrame,
) -> pd.DataFrame:
    """This method calculates the mean and median of the signal dataframe
    over the different stimulus repetitions.

    Parameters
    ----------
    signal : pd.DataFrame
        The signal dataframe containing the data of the ROI. If the direction
        is pooled, the dataframe contains the data of all directions and the
        mean and median are calculated over all directions. If the direction
        is not pooled, the dataframe contains the data of the selected
        direction and the mean and median are calculated over it.
        These computations are done in the `sf_tf_grid` callback.
    Returns
    -------
    pd.DataFrame
        The signal dataframe containing the mean and median of the signal
        appended to the original dataframe at the end.
    """
    mean_df = (
        signal.groupby(["sf", "tf", "stimulus_frames"])
        .agg({"signal": "mean"})
        .reset_index()
    )
    mean_df["stimulus_repetition"] = "mean"
    combined_df = pd.concat([signal, mean_df], ignore_index=True)

    median_df = (
        signal.groupby(["sf", "tf", "stimulus_frames"])
        .agg({"signal": "median"})
        .reset_index()
    )
    median_df["stimulus_repetition"] = "median"
    combined_df = pd.concat([combined_df, median_df], ignore_index=True)

    return combined_df


def find_peak_coordinates(
    fitted_gaussian_matrix: np.ndarray,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    matrix_dimension: int,
) -> tuple:
    """This method finds the peak coordinates of the fitted gaussian matrix.
    The gausisan matrix is the result of the fitting process of the two
    dimensional gaussian (described by Andermnn et al. 2011.) to the
    sampled data (the median subtracted response matrix).
    Here we are interested in finding the peak coordinates of the fitted
    gaussian as it represents the theoretical preferred spatial and temporal
    frequency of the neuron.

    Parameters
    ----------
    fitted_gaussian_matrix : np.ndarray
        The fitted gaussian matrix obtained from the precalculated fits.
    spatial_frequencies : np.ndarray
        The spatial frequencies that are used in the experiment.
    temporal_frequencies : np.ndarray
        The temporal frequencies that are used in the experiment.
    matrix_dimension : int
        The matrix definition used to generate the fitted_gaussian_matrix.

    Returns
    -------
    tuple
        The peak coordinates of the fitted gaussian matrix.
    """
    peak_indices = np.unravel_index(
        np.argmax(fitted_gaussian_matrix), fitted_gaussian_matrix.shape
    )

    spatial_freq_linspace = np.linspace(
        spatial_frequencies.min(),
        spatial_frequencies.max(),
        matrix_dimension,
    )
    temporal_freq_linspace = np.linspace(
        temporal_frequencies.min(),
        temporal_frequencies.max(),
        matrix_dimension,
    )

    sf = spatial_freq_linspace[peak_indices[0]]
    tf = temporal_freq_linspace[peak_indices[1]]
    return tf, sf


def call_get_gaussian_matrix_to_be_plotted(
    n_roi: int,
    fit_outputs: dict,
    spatial_frequencies: np.ndarray,
    temporal_frequencies: np.ndarray,
    matrix_dimension: int,
) -> dict:
    """This method is a wrapper for the get_gaussian_matrix_to_be_plotted
    method that iterates over all the ROIs.

    Parameters
    ----------
    n_roi : int
        The number of ROIs.
    fit_outputs : dict
        The fit outputs obtained from the precalculated fits.
    spatial_frequencies : np.ndarray
        The spatial frequencies that are used in the experiment.
    temporal_frequencies : np.ndarray
        The temporal frequencies that are used in the experiment.
    matrix_dimension : int
        The matrix definition used to generate the fitted_gaussian_matrix.

    Returns
    -------
    dict
        The fitted gaussian matrix obtained from the precalculated fits.
    """
    fitted_gaussian_matrix = {}

    for roi_id in range(n_roi):
        fitted_gaussian_matrix[
            (roi_id, "pooled")
        ] = get_gaussian_matrix_to_be_plotted(
            kind="custom",
            roi_id=roi_id,
            fit_output=fit_outputs,
            sfs=np.asarray(spatial_frequencies),
            tfs=np.asarray(temporal_frequencies),
            matrix_dimension=matrix_dimension,
            direction="pooled",
        )

    return fitted_gaussian_matrix
