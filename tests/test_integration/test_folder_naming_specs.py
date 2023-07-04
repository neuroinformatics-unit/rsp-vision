import pytest

from rsp_vision.objects import folder_naming_specs


# Tests
def test_FolderNamingSpecs_constructor(experimental_folders, config):
    for fs in experimental_folders:
        folder_naming_specs.FolderNamingSpecs(fs.folder, config)


def test_FolderNamingSpecs_constructor_fails(config):
    with pytest.raises(Exception):
        folder_naming_specs.FolderNamingSpecs("AS_1111877", config)
