import numpy as np
import pandas as pd
import pytest

from rsp_vision.analysis.spatial_freq_temporal_freq import (
    FrequencyResponsiveness,
)


@pytest.fixture
def responsiveness():
    return FrequencyResponsiveness.__new__(FrequencyResponsiveness)


@pytest.fixture
def median_subtracted_response():
    data = {
        "sf": [0.03, 0.03, 0.03],
        "tf": [0.1, 0.2, 0.3],
        "subtracted": [1.0, 2.0, 3.0],
    }
    df = pd.DataFrame(data)
    df = df.set_index(["sf", "tf"])

    return df


def test_get_preferred_sf_tf(responsiveness, median_subtracted_response):
    # Call the function
    result = responsiveness.get_preferred_sf_tf(median_subtracted_response)

    # Assert that the result is a tuple
    assert isinstance(result, tuple)

    # Assert that the tuple has three elements
    assert len(result) == 3

    # Assert that the elements are of type float
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert isinstance(result[2], float)

    # Assert that the elements are within expected values
    expected_sf = 0.03
    expected_tf = 0.3
    expected_value = 3.0
    assert result[0] == expected_sf
    assert result[1] == expected_tf
    assert result[2] == expected_value


def test_get_median_subtracted_response(responsiveness):
    # Define input data
    data = {
        "roi_id": [1, 1, 1, 2, 2, 2],
        "direction": [0, 1, 2, 0, 1, 2],
        "sf": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
        "tf": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        "subtracted": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    df = pd.DataFrame(data)

    # Call the function
    result = responsiveness.get_median_subtracted_response(df, 1, 0)

    # Assert that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert that the result has the expected columns
    expected_columns = ["subtracted"]
    expected_indices_names = ["sf", "tf"]
    assert all(column in result.columns for column in expected_columns)
    assert all(
        index_name in result.index.names
        for index_name in expected_indices_names
    )

    # Assert that the result has the expected number of rows
    expected_num_rows = 1
    assert len(result) == expected_num_rows

    # Assert that the response values are median-subtracted
    expected_response = [1.0]
    assert all(
        response == expected_response[i]
        for i, response in enumerate(result["subtracted"])
    )


def test_get_median_subtracted_response_2d_matrix(
    responsiveness, median_subtracted_response
):
    sfs = np.array([0.03])
    tfs = np.array([0.1, 0.2, 0.3])

    # Call the function
    result = responsiveness.get_median_subtracted_response_2d_matrix(
        median_subtracted_response, sfs, tfs
    )

    # Assert that the result is a numpy array
    assert isinstance(result, np.ndarray)

    # Assert that the result has the expected shape
    expected_shape = (len(sfs), len(tfs))
    assert result.shape == expected_shape

    # Assert that the response values are median-subtracted
    expected_response = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )
    assert np.allclose(result, expected_response)
