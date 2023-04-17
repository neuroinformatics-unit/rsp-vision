import pathlib
import pickle
from functools import lru_cache

import pytest

from tests.test_integration.generate_mock_data import (
    get_data_raw_object_mock,
    get_experimental_variables_mock,
    get_photon_data_mock,
    get_response_mock,
)


@pytest.fixture
def experimental_variables():
    return get_experimental_variables_mock()


@pytest.fixture
def data_raw():
    return get_data_raw_object_mock()


@pytest.fixture
def photon_data():
    return get_photon_data_mock()


@pytest.fixture
def response():
    @lru_cache(maxsize=None)  # Cache all computed responses
    def _responses(seed_number):
        return get_response_mock(seed_number)

    return _responses


@pytest.fixture
def expected_outputs():
    path = pathlib.Path(__file__).parent.absolute()
    output_path = path / "test_integration" / "mock_data" / "outputs.plk"
    with open(output_path, "rb") as f:
        outputs = pickle.load(f)
    return outputs
