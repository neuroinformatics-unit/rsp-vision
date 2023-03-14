import numpy as np


def test_get_response_and_baseline_windows(
    get_variables, get_freq_response_instance
):
    (
        _,
        n_roi,
        _,
        n_stim,
        _,
        _,
    ) = get_variables

    response = get_freq_response_instance
    (
        window_mask_response,
        window_mask_baseline,
    ) = response.get_response_and_baseline_windows()

    assert (
        len(window_mask_baseline)
        == len(window_mask_response)
        == (n_stim * n_roi)
    )


def test_calculate_mean_response_and_baseline(get_freq_response_instance):
    response = get_freq_response_instance
    response.calculate_mean_response_and_baseline()

    # based on random seed = 101
    assert int(response.data.responses["subtracted"].values[1]) == 34


def test_nonparam_anova_over_rois(get_freq_response_instance):
    response = get_freq_response_instance
    response.calculate_mean_response_and_baseline()
    p_values = response.nonparam_anova_over_rois()

    decimal_points = 3
    p_values = np.around(
        np.fromiter(p_values.values(), dtype=float), decimal_points
    )
    # based on random seed = 101
    p_values_seed_101 = np.array([0.055, 0.473, 0.324, 0.127, 0.653])

    assert np.all(p_values == p_values_seed_101)


def test_perform_sign_tests(get_freq_response_instance):
    response = get_freq_response_instance
    response.calculate_mean_response_and_baseline()
    p_st, p_wsrt = response.perform_sign_tests()

    decimal_points = 3
    p_st = np.around(np.fromiter(p_st.values(), dtype=float), decimal_points)
    p_wsrt = np.around(
        np.fromiter(p_wsrt.values(), dtype=float), decimal_points
    )

    # based on random seed = 101
    p_st_seed_101 = np.array([0.968, 0.924, 0.032, 0.271, 0.846])
    p_wsrt_seed_101 = np.array([0.855, 0.928, 0.18, 0.195, 0.55])

    assert np.all(p_st == p_st_seed_101)
    assert np.all(p_wsrt == p_wsrt_seed_101)


def test_response_magnitude(get_freq_response_instance):
    response = get_freq_response_instance
    response.calculate_mean_response_and_baseline()

    magnitude = response.response_magnitude()["magnitude"]

    decimal_points = 3
    magnitude = np.around(np.fromiter(magnitude, dtype=float), decimal_points)
    # based on random seed = 101
    magnitude_seed_101 = np.array(
        [
            0.2,
            -0.118,
            0.163,
            -0.282,
            -0.179,
            0.17,
            -0.428,
            -0.364,
            -0.238,
            0.356,
            0.028,
            0.089,
            0.698,
            -0.224,
            0.302,
            -0.049,
            0.014,
            0.346,
            -0.315,
            -0.062,
        ]
    )

    assert np.all(magnitude == magnitude_seed_101)


def test_find_significant_rois(get_freq_response_instance):
    response = get_freq_response_instance
    response()

    assert len(response.data.responsive_rois) == 0
