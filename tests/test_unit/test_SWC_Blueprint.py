from pathlib import Path

from rsp_vision.objects.SWC_Blueprint import (
    SessionFolder,
    SubjectFolder,
)


def test_SWC_Blueprint_Spec(tmp_path, blueprint_spec):
    Path(blueprint_spec.path).mkdir(parents=True, exist_ok=True)

    assert (tmp_path / "my_project").exists()


def test_SubjectFolder_with_folder_naming_specs(tmp_path, blueprint_spec, folder_naming_specs):
    for fns in folder_naming_specs:
        subject_folder = SubjectFolder(blueprint_spec, fns, 0)
        Path(subject_folder.sub_folder_path).mkdir(parents=True, exist_ok=True)

        assert (
            tmp_path
            / "my_project"
            / "derivatives"
            / subject_folder.sub_folder_name
        ).exists()

        assert (
            subject_folder.sub_folder_name
            == f"sub-000_line-{fns.mouse_line}_id-{fns.mouse_id}"
        )

def test_SubjectFolder_with_table_row(tmp_path, blueprint_spec, table_row):
    subject_folder = SubjectFolder(blueprint_spec, table_row, 0)
    Path(subject_folder.sub_folder_path).mkdir(parents=True, exist_ok=True)

    assert (
        tmp_path
        / "my_project"
        / "derivatives"
        / subject_folder.sub_folder_name
    ).exists()

    assert (
        subject_folder.sub_folder_name
        == f"sub-{table_row['sub']}_line-{table_row['mouse line']}_id-{table_row['mouse id']}"
    )

def test_SessionFolder_folder_naming_specs(tmp_path, blueprint_spec, folder_naming_specs):
    for fns in folder_naming_specs:
        subject_folder = SubjectFolder(blueprint_spec, fns, 0)
        session_folder = SessionFolder(subject_folder, fns, 0)
        Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

        assert (
            tmp_path
            / "my_project"
            / "derivatives"
            / subject_folder.sub_folder_name
            / session_folder.ses_folder_name
        ).exists()

        monitor = fns.monitor_position[8:].replace("_", "-").replace("-", "")

        test_name = (
            f"ses-000_hemisphere-{fns.hemisphere}_"
            + f"region-{fns.brain_region}_monitor-{monitor}"
        )

        if (fns.fov is None) and (fns.cre is None):
            assert session_folder.ses_folder_name == test_name
        elif (fns.fov is not None) and (fns.cre is None):
            assert (
                session_folder.ses_folder_name == test_name + f"_fov-{fns.fov}"
            )
        elif (fns.fov is None) and (fns.cre is not None):
            assert (
                session_folder.ses_folder_name == test_name + f"_cre-{fns.cre}"
            )
        else:
            assert (
                session_folder.ses_folder_name
                == test_name + f"_fov-{fns.fov}_cre-{fns.cre}"
            )

def test_SessionFolder_table_row(tmp_path, blueprint_spec, table_row):
    subject_folder = SubjectFolder(blueprint_spec, table_row, 0)
    session_folder = SessionFolder(subject_folder, table_row, 0)
    Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

    assert (
        tmp_path
        / "my_project"
        / "derivatives"
        / subject_folder.sub_folder_name
        / session_folder.ses_folder_name
    ).exists()

    test_name = (
        f"ses-{table_row['ses']}_hemisphere-{table_row['hemisphere']}_"
        + f"region-{table_row['brain region']}_monitor-{table_row['monitor position']}"
    )

    if (table_row["fov"] is None) and (table_row["cre"] is None):
        assert session_folder.ses_folder_name == test_name
    elif (table_row["fov"] is not None) and (table_row["cre"] is None):
        assert (
            session_folder.ses_folder_name == test_name + f"_fov-{table_row['fov']}"
        )
    elif (table_row["fov"] is None) and (table_row["cre"] is not None):
        assert (
            session_folder.ses_folder_name == test_name + f"_cre-{table_row['cre']}"
        )
    else:
        assert (
            session_folder.ses_folder_name
            == test_name + f"_fov-{table_row['fov']}_cre-{table_row['cre']}"
        )