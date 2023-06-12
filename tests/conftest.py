import pathlib

import pandas as pd
import pytest

from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs
from tests.test_integration.generate_mock_data import (
    get_config_mock,
    get_data_raw_object_mock,
    get_experimental_variables_mock,
    get_photon_data_mock,
    get_response_mock,
)


@pytest.fixture
def experimental_folders(tmp_path):
    class FolderStructure:
        def __init__(self, parent_folder: str, folder: str):
            self.parent_folder = parent_folder
            self.folder = folder

    folder_test_list = [
        FolderStructure("AS_1111877", "AS_1111877_hL_V1_monitor_front"),
        FolderStructure("AS_130_3", "AS_130_3_hL_V1_monitor_right"),
        FolderStructure("BY_317_2", "BY_317_2_hL_RSPd_monitor_left"),
        FolderStructure(
            "BY_IAA_1117275", "BY_IAA_1117275_hL_RSPg_monitor_front"
        ),
        FolderStructure("CX_79_2", "CX_79_2_hL_RSPd_monitor_right"),
        FolderStructure("CX_102_2", "CX_102_2_hL_RSPd_FOV3c_monitor_front"),
        FolderStructure("CX_1111923", "CX_1111923_hL_V1_monitor_front-right"),
        FolderStructure(
            "SG_1118210", "SG_1118210_hR_RSPd_cre-off_monitor_front"
        ),
    ]

    for fs in folder_test_list:
        pathlib.Path(f"{tmp_path}/{fs.parent_folder}/{fs.folder}").mkdir(
            parents=True, exist_ok=True
        )

    return folder_test_list


@pytest.fixture
def folder_naming_specs(experimental_folders, config):
    fns = []
    for folder in experimental_folders:
        fns.append(FolderNamingSpecs(folder.folder, config))
    return fns


@pytest.fixture
def config():
    return get_config_mock()


@pytest.fixture
def one_day_objects():
    return (
        get_photon_data_mock(),
        get_experimental_variables_mock(),
        get_data_raw_object_mock(),
    )


@pytest.fixture
def multiple_days_objects():
    return (
        get_photon_data_mock(multiple_days=True),
        get_experimental_variables_mock(multiple_days=True),
        get_data_raw_object_mock(multiple_days=True),
    )


@pytest.fixture
def variables():
    return get_experimental_variables_mock()


@pytest.fixture
def var_mult_days():
    return get_experimental_variables_mock(multiple_days=True)


@pytest.fixture
def response():
    def _responses(seed_number, multiple_days=False):
        return get_response_mock(seed_number, multiple_days=multiple_days)

    return _responses


@pytest.fixture
def expected_outputs():
    path = pathlib.Path(__file__).parent.absolute()
    output_path = path / "test_integration" / "mock_data" / "outputs.plk"
    with open(output_path, "rb") as f:
        outputs = pd.read_pickle(f)
    return outputs
