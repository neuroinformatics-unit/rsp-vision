import numpy as np
import pandas as pd
import pytest
from itertools import chain


seeds = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]


def test_get_response_and_baseline_windows(experimental_variables, response):
    (
        _,
        n_roi,
        _,
        n_stim,
        _,
        _,
    ) = experimental_variables

    _response = response(1)
    (
        window_mask_response,
        window_mask_baseline,
    ) = _response.get_response_and_baseline_windows()

    assert (
        len(window_mask_baseline)
        == len(window_mask_response)
        == (n_stim * n_roi)
    )


@pytest.mark.parametrize("seed", seeds)
def test_calculate_mean_response_and_baseline(
    response, expected_outputs, seed
):
    _response = response(seed)
    _response.calculate_mean_response_and_baseline()

    outputs = expected_outputs[str(seed)]

    assert np.all(
        np.around(
            np.fromiter(
                _response.data.responses["subtracted"].values, dtype=float
            ),
            decimals=3,
        )
        == np.around(
            np.fromiter(
                outputs["responses"]["subtracted"].values(), dtype=float
            ),
            decimals=3,
        )
    )


@pytest.mark.parametrize("seed", seeds)
def test_nonparam_anova_over_rois(response, expected_outputs, seed):
    _response = response(seed)
    _response.calculate_mean_response_and_baseline()
    p_values = _response.nonparam_anova_over_rois()

    decimal_points = 3
    p_values = np.around(
        np.fromiter(p_values.values(), dtype=float), decimal_points
    )
    outputs = expected_outputs[str(seed)]
    p_values_expected = np.around(
        np.fromiter(
            outputs["p_values"]["Kruskal-Wallis test"].values, dtype=float
        ),
        decimal_points,
    )
    assert np.all(
        p_values == p_values_expected
    )


@pytest.mark.parametrize("seed", seeds)
def test_perform_sign_tests(response, expected_outputs, seed):
    _response = response(seed)
    _response.calculate_mean_response_and_baseline()
    p_st, p_wsrt = _response.perform_sign_tests()

    decimal_points = 3
    p_st = np.around(np.fromiter(p_st.values(), dtype=float), decimal_points)
    p_wsrt = np.around(
        np.fromiter(p_wsrt.values(), dtype=float), decimal_points
    )

    outputs = expected_outputs[str(seed)]
    p_st_expected = np.around(
        np.fromiter(outputs["p_values"]["Sign test"].values, dtype=float), decimal_points
    )
    p_wsrt_expected = np.around(
        np.fromiter(outputs["p_values"]["Wilcoxon signed rank test"].values, dtype=float), decimal_points
    )

    assert np.all(p_st == p_st_expected)
    assert np.all(p_wsrt == p_wsrt_expected)


@pytest.mark.parametrize("seed", seeds)
def test_response_magnitude(response, expected_outputs, seed):
    _response = response(seed)
    _response.calculate_mean_response_and_baseline()

    magnitude = _response.response_magnitude()["magnitude"]

    decimal_points = 3
    magnitude = np.around(np.fromiter(magnitude, dtype=float), decimal_points)
    outputs = expected_outputs[str(seed)]
    magnitude_expected = np.around(
        np.fromiter(outputs["magnitude_over_medians"]["magnitude"].values(), dtype=float), decimal_points
    )

    assert np.all(magnitude == magnitude_expected)


@pytest.mark.parametrize("seed", seeds)
def test_find_significant_rois(response, expected_outputs, seed):
    _response = response(seed)
    _response.calculate_mean_response_and_baseline()
    p_values = pd.DataFrame()
    p_values["Kruskal-Wallis test"] = _response.nonparam_anova_over_rois()
    magnitude = _response.response_magnitude()

    significant_rois = _response.find_significant_rois(p_values, magnitude)

    outputs = expected_outputs[str(seed)]
    significant_rois_expected = set(outputs["responsive_rois"])

    assert significant_rois == significant_rois_expected
