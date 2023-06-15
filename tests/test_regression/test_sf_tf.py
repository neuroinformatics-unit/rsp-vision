# these tests require complex fixtures which are defined in
# tests/conftest.py. The fixtures are used to generate mock data
# which is used to test the full functionality of the
# FrequencyResponsiveness class. Their output is compared to
# expected outputs which are stored in a pickle file.


import numpy as np
import pandas as pd
import pytest

seeds = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]


def test_get_response_and_baseline_windows(
    variables, var_mult_days, response, n_roi
):
    # using any seed, it does not matter for this test
    for v in [variables, var_mult_days]:
        if v.n_days == 1:
            _response = response(1)
        else:
            _response = response(1, multiple_days=True)
        (
            window_mask_response,
            window_mask_baseline,
        ) = _response.get_response_and_baseline_windows()

        assert (
            len(window_mask_baseline)
            == len(window_mask_response)
            == (v.n_stim * n_roi)
        )


@pytest.mark.parametrize("seed", seeds)
def test_calculate_mean_response_and_baseline(
    response, expected_outputs, seed
):
    if seed < 11:
        _response = response(seed)
    else:
        _response = response(seed, multiple_days=True)

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
    if seed < 11:
        _response = response(seed)
    else:
        _response = response(seed, multiple_days=True)

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
    assert np.all(p_values == p_values_expected)


@pytest.mark.parametrize("seed", seeds)
def test_perform_sign_tests(response, expected_outputs, seed):
    if seed < 11:
        _response = response(seed)
    else:
        _response = response(seed, multiple_days=True)

    _response.calculate_mean_response_and_baseline()
    p_st, p_wsrt = _response.perform_sign_tests()

    decimal_points = 3
    p_st = np.around(np.fromiter(p_st.values(), dtype=float), decimal_points)
    p_wsrt = np.around(
        np.fromiter(p_wsrt.values(), dtype=float), decimal_points
    )

    outputs = expected_outputs[str(seed)]
    p_st_expected = np.around(
        np.fromiter(outputs["p_values"]["Sign test"].values, dtype=float),
        decimal_points,
    )
    p_wsrt_expected = np.around(
        np.fromiter(
            outputs["p_values"]["Wilcoxon signed rank test"].values,
            dtype=float,
        ),
        decimal_points,
    )

    assert np.all(p_st == p_st_expected)
    assert np.all(p_wsrt == p_wsrt_expected)


@pytest.mark.parametrize("seed", seeds)
def test_response_magnitude(response, expected_outputs, seed):
    if seed < 11:
        _response = response(seed)
    else:
        _response = response(seed, multiple_days=True)

    _response.calculate_mean_response_and_baseline()

    magnitude = _response.response_magnitude()["magnitude"]

    decimal_points = 3
    magnitude = np.around(np.fromiter(magnitude, dtype=float), decimal_points)
    outputs = expected_outputs[str(seed)]
    magnitude_expected = np.around(
        np.fromiter(
            outputs["magnitude_over_medians"]["magnitude"].values(),
            dtype=float,
        ),
        decimal_points,
    )

    assert np.all(magnitude == magnitude_expected)


@pytest.mark.parametrize("seed", seeds)
def test_find_significant_rois(response, expected_outputs, seed):
    if seed < 11:
        _response = response(seed)
    else:
        _response = response(seed, multiple_days=True)

    _response.calculate_mean_response_and_baseline()
    p_values = pd.DataFrame()
    p_values["Kruskal-Wallis test"] = _response.nonparam_anova_over_rois()
    magnitude = _response.response_magnitude()

    significant_rois = _response.find_significant_rois(p_values, magnitude)

    outputs = expected_outputs[str(seed)]
    significant_rois_expected = set(outputs["responsive_rois"])

    assert significant_rois == significant_rois_expected


@pytest.mark.parametrize("seed", seeds)
def test_get_gaussian_fits_for_roi(response, expected_outputs, seed):
    if seed < 11:
        _response = response(seed)
    else:
        _response = response(seed, multiple_days=True)

    _response()
    outputs = expected_outputs[str(seed)]

    for roi_id in range(_response.data.n_roi):
        for dir in _response.data.directions:
            measured_preference = outputs["measured_preference"][(roi_id, dir)]
            median_subtracted_response = outputs["median_subtracted_response"][
                (roi_id, dir)
            ]
            # we do not test the fit output because it will change
            # every time even if the seed is the same

            assert np.all(
                np.around(
                    _response.data.measured_preference[(roi_id, dir)],
                    decimals=3,
                )
                == np.around(measured_preference, decimals=3)
            )

            assert np.all(
                np.around(
                    _response.data.median_subtracted_response[(roi_id, dir)],
                    decimals=3,
                )
                == np.around(median_subtracted_response, decimals=3)
            )
