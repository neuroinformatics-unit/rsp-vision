#  these tests have their own small set of fixtures which
#  are defined in this file. They are simpler than the one used in
#  test_sf_tf.py because they only test the static methods of the
#  FrequencyResponsiveness class. The static methods do not require
#  the full complexity of the fixtures used in test_sf_tf.py


import numpy as np
import pandas as pd
import pytest

from rsp_vision.analysis.spatial_freq_temporal_freq import (
    FrequencyResponsiveness,
)


@pytest.fixture
def responsiveness():
    return FrequencyResponsiveness.__new__(FrequencyResponsiveness)


def test_get_median_subtracted_response_and_params(responsiveness):
    responses = pd.DataFrame(
        {
            "subtracted": [1, 2, 3, 4],
            "sf": [0.1, 0.1, 0.2, 0.2],
            "tf": [0.1, 0.2, 0.1, 0.2],
            "roi_id": [1, 1, 1, 1],
            "direction": [0, 0, 0, 0],
        }
    )

    (
        median_subtracted_response_2d_matrix,
        sf_0,
        tf_0,
        peak_response,
    ) = responsiveness.get_median_subtracted_response_and_params(
        responses=responses,
        roi_id=1,
        dir=0,
        sfs=np.array([0.1, 0.2]),
        tfs=np.array([0.1, 0.2]),
    )

    assert median_subtracted_response_2d_matrix.shape == (2, 2)
    assert np.all(
        median_subtracted_response_2d_matrix == np.array([[1, 2], [3, 4]])
    )
    assert sf_0 == 0.2
    assert tf_0 == 0.2
    assert peak_response == 4
