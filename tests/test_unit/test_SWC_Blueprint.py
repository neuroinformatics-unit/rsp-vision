from pathlib import Path
import pytest
from rsp_vision.objects.SWC_Blueprint import (
    SWC_Blueprint_Spec,
    SubjectFolder,
    SessionFolder,
)


@pytest.fixture
def blueprint_spec(tmp_path):
    spec = SWC_Blueprint_Spec(
        project_name="my_project",
        raw_data=True,
        derivatives=True,
        local_path=tmp_path,
    )
    return spec


def test_SWC_Blueprint_Spec(tmp_path, blueprint_spec):
    Path(blueprint_spec.path).mkdir(parents=True, exist_ok=True)

    assert (tmp_path / "my_project").exists()


def test_SubjectFolder(tmp_path, blueprint_spec, folder_naming_specs):
    for fns in folder_naming_specs:
        subject_folder = SubjectFolder(blueprint_spec, fns, 0)
        Path(subject_folder.sub_folder_path).mkdir(parents=True, exist_ok=True)

        assert (
            tmp_path / "my_project" / "derivatives" / subject_folder.sub_folder_name
        ).exists()

        assert (
            subject_folder.sub_folder_name
            == f"sub-000_line-{fns.mouse_line}_id-{fns.mouse_id}"
        )


def test_SessionFolder(tmp_path, blueprint_spec, folder_naming_specs):
    for fns in folder_naming_specs:
        subject_folder = SubjectFolder(blueprint_spec, fns, 0)
        session_folder = SessionFolder(subject_folder, fns, 0)
        Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

        assert (
            tmp_path / "my_project" / "derivatives" / subject_folder.sub_folder_name / session_folder.ses_folder_name
        ).exists()

        monitor = fns.monitor_position[8:].replace("_", "-").replace("-", "")

        test_name = f"ses-000_hemisphere-{fns.hemisphere}_region-{fns.brain_region}_monitor-{monitor}"

        if (fns.fov is None) and (fns.cre is None):
            assert (
                session_folder.ses_folder_name
                == test_name
            )
        elif (fns.fov is not None) and (fns.cre is None):
            assert (
                session_folder.ses_folder_name
                == test_name + f"_fov-{fns.fov}"
            )
        elif (fns.fov is None) and (fns.cre is not None):
            assert (
                session_folder.ses_folder_name
                == test_name + f"_cre-{fns.cre}"
            )
        else:
            assert (
                session_folder.ses_folder_name
                == test_name + f"_fov-{fns.fov}_cre-{fns.cre}"
            )
