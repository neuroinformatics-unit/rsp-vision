import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from rsp_vision.analysis.gaussians_calculations import (
    create_gaussian_matrix,
    elliptical_gaussian_andermann,
    fit_2D_gaussian_to_data,
    get_gaussian_matrix_to_be_plotted,
    single_fit,
)


@pytest.fixture
def parameters_to_fit():
    params = np.array([10, 0.01, 0.1, 0.005, 0.05, 1])
    sfs = np.array([0.01, 0.02, 0.03])
    tfs = np.array([0.1, 0.2, 0.3])
    response_matrix = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    lower_bounds = np.array([-10, -10, -10, -10, -10, -10])
    upper_bounds = np.array([100, 100, 100, 100, 100, 100])

    return params, sfs, tfs, response_matrix, lower_bounds, upper_bounds


def test_elliptical_gaussian_andermann():
    # Define inputs
    peak_response = 1.0
    sf = 0.1
    tf = 0.2
    sf_0 = 0.05
    tf_0 = 0.1
    sigma_sf = 0.01
    sigma_tf = 0.02
    𝜻_power_law_exp = 0.5

    # Expected output
    expected_output = 0.9999000049998332

    # Call the method
    output = elliptical_gaussian_andermann(
        peak_response, sf, tf, sf_0, tf_0, sigma_sf, sigma_tf, 𝜻_power_law_exp
    )

    # Assert the output is correct
    assert np.allclose(output, expected_output, atol=1e-3)


def test_single_fit(parameters_to_fit):
    # Define inputs
    (
        params,
        sfs,
        tfs,
        response_matrix,
        lower_bounds,
        upper_bounds,
    ) = parameters_to_fit

    # Call the function
    result = single_fit(
        params, sfs, tfs, response_matrix, lower_bounds, upper_bounds
    )

    # Assert that the result is an instance of OptimizeResult
    assert isinstance(result, OptimizeResult)

    # Assert that the result has the expected attributes
    expected_attributes = ["x"]
    for attribute in expected_attributes:
        assert hasattr(
            result, attribute
        ), f"Expected attribute {attribute} not found in result"

    # Assert that the optimized parameters are within bounds
    assert np.all(
        result.x >= lower_bounds
    ), "Optimized parameters below lower bounds"
    assert np.all(
        result.x <= upper_bounds
    ), "Optimized parameters above upper bounds"


def test_fit_2D_gaussian_to_data(parameters_to_fit):
    # Define inputs
    (
        params,
        sfs,
        tfs,
        response_matrix,
        lower_bounds,
        upper_bounds,
    ) = parameters_to_fit
    config = {
        "fitting": {
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "jitter": 0.1,
            "iterations_to_fit": 3,
        }
    }

    # Call the function
    result = fit_2D_gaussian_to_data(sfs, tfs, response_matrix, params, config)

    # Assert that the result is an instance of OptimizeResult
    assert isinstance(result, OptimizeResult)

    # Assert that the result has the expected attributes
    expected_attributes = ["x"]
    for attribute in expected_attributes:
        assert hasattr(
            result, attribute
        ), f"Expected attribute {attribute} not found in result"

    # Assert that the optimized parameters are within bounds
    assert np.all(
        result.x >= config["fitting"]["lower_bounds"]
    ), "Optimized parameters below lower bounds"
    assert np.all(
        result.x <= config["fitting"]["upper_bounds"]
    ), "Optimized parameters above upper bounds"


def test_create_gaussian_matrix(parameters_to_fit):
    # Define inputs
    params, sfs, tfs, _, _, _ = parameters_to_fit

    # Call the function
    result = create_gaussian_matrix(params, sfs, tfs)

    # Assert that the result is a 2D array
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2

    # Assert that the result has the expected shape
    expected_shape = (len(sfs), len(tfs))
    assert result.shape == expected_shape

    # Assert that the result is within expected values
    expected_result = np.array(
        [
            [10.0, 9.98750781, 9.96864792],
            [9.98738297, 9.999875, 9.9955987],
            [9.9683349, 9.99540978, 9.99968599],
        ]
    )
    assert np.allclose(result, expected_result, atol=1e-3)


def test_gaussian_matrix_for_6x6_matrix_single_direction():
    # Test case for a 6x6 matrix with a single direction
    kind = "6x6 matrix"
    roi_id = 1
    direction = 0
    fit_output = {(roi_id, direction): np.asarray([6, 5, 4, 3, 2, 1])}
    sfs = np.array([0.1, 0.2, 0.3])
    tfs = np.array([0.4, 0.5, 0.6])
    pooled_directions = False
    expected_matrix = np.array(
        [
            [6.981e-67, 2.853e-68, 1.538e-69],
            [1.301e-43, 1.928e-44, 2.978e-45],
            [1.287e-32, 4.050e-33, 1.157e-33],
        ]
    )

    matrix = get_gaussian_matrix_to_be_plotted(
        kind,
        roi_id,
        fit_output,
        sfs,
        tfs,
        pooled_directions,
        direction,
    )

    assert np.allclose(matrix, expected_matrix)
