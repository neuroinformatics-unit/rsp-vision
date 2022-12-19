from pathlib import Path

import pytest

from load_suite2p.objects import folder_naming_specs

# Mocks


class FolderStructure:
    def __init__(self, parent_folder: str, folder: str):
        self.parent_folder = parent_folder
        self.folder = folder


folder_test_list = [
    FolderStructure("AS_1111877", "AS_1111877_hL_V1_monitor_front"),
    FolderStructure("AS_130_3", "AS_130_3_hL_V1_monitor_right"),
    FolderStructure("BY_317_2", "BY_317_2_hL_RSPd_monitor_left"),
    FolderStructure("BY_IAA_1117275", "BY_IAA_1117275_hL_RSPg_monitor_front"),
    FolderStructure("CX_79_2", "CX_79_2_hL_RSPd_monitor_right"),
    FolderStructure("CX_102_2", "CX_102_2_hL_RSPd_FOV3c_monitor_front"),
    FolderStructure("CX_1111923", "CX_1111923_hL_V1_monitor_front-right"),
    FolderStructure("SG_1118210", "SG_1118210_hR_RSPd_cre-off_monitor_front"),
]

config = {
    "parser": "Parser2pRSP",
    "paths": {"imaging": "test_data/"},
}

for fs in folder_test_list:
    Path(f"test_data/{fs.parent_folder}/{fs.folder}").mkdir(
        parents=True, exist_ok=True
    )


# Tests
def test_FolderNamingSpecs_constructor():
    for fs in folder_test_list:
        folder_naming_specs.FolderNamingSpecs(fs.folder, config)


def test_FolderNamingSpecs_constructor_fails():
    with pytest.raises(Exception):
        folder_naming_specs.FolderNamingSpecs("AS_1111877", config)
