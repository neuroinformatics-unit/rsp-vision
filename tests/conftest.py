import pathlib

import pandas as pd
import pytest

from tests.fixtures_helpers import (
    get_data_raw_object_mock,
    get_photon_data_mock,
    get_response_mock,
    get_shared_variables_to_generate_mock_data,
    make_variables_day_related,
)

@pytest.fixture
def general_variables():
    _, _, params = get_shared_variables_to_generate_mock_data()
    return params

@pytest.fixture
def one_day_objects(variables):
    return (
        get_photon_data_mock(),
        variables,
        get_data_raw_object_mock(),
    )


@pytest.fixture
def multiple_days_objects(var_mult_days):
    return (
        get_photon_data_mock(multiple_days=True),
        var_mult_days,
        get_data_raw_object_mock(multiple_days=True),
    )


@pytest.fixture
def variables(general_variables):
    return make_variables_day_related(general_variables)


@pytest.fixture
def n_roi(general_variables):
    return general_variables.n_roi


@pytest.fixture
def var_mult_days(general_variables):
    return make_variables_day_related(general_variables, multiple_days=True)


@pytest.fixture
def response():
    def _responses(seed_number, multiple_days=False):
        return get_response_mock(seed_number, multiple_days=multiple_days)

    return _responses


@pytest.fixture
def expected_outputs():
    path = pathlib.Path(__file__).parent.absolute()
    output_path = path / "test_regression" / "mock_data" / "outputs.plk"
    with open(output_path, "rb") as f:
        outputs = pd.read_pickle(f)
    return outputs
