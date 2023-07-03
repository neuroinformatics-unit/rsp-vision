import pathlib

import pandas as pd
import pytest

from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs
from rsp_vision.objects.SWC_Blueprint import (
    SessionFolder,
    SubjectFolder,
    SWC_Blueprint_Spec,
)
from tests.fixtures_helpers import (
    get_config_mock,
    get_data_raw_object_mock,
    get_photon_data_mock,
    get_response_mock,
    get_shared_variables_to_generate_mock_data,
    make_variables_day_related,
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
def table_row():
    return {
        "sub": "000",
        "ses": "000",
        "mouse id": "1111877",
        "mouse line": "hL_V1",
        "hemisphere": "left",
        "brain region": "V1",
        "monitor position": "front",
        "cre": "off",
        "fov": "3c",
    }
    


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
def blueprint_spec(tmp_path):
    spec = SWC_Blueprint_Spec(
        project_name="my_project",
        raw_data=True,
        derivatives=True,
        local_path=tmp_path,
    )
    return spec


@pytest.fixture
def one_folder_naming_specs(folder_naming_specs):
    return folder_naming_specs[0]


@pytest.fixture
def subject_folder(blueprint_spec, one_folder_naming_specs):
    return SubjectFolder(blueprint_spec, one_folder_naming_specs, 4)


@pytest.fixture
def session_folder(subject_folder, one_folder_naming_specs):
    return SessionFolder(subject_folder, one_folder_naming_specs, 0)


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
def photon_data():
    # full photon data, as it becomes after the data is processed
    response = get_response_mock(0)
    response()
    return response.data


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
