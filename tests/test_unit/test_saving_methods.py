from pathlib import Path

import pandas as pd

from rsp_vision.save.save_data import (
    get_sub_and_ses,
    save_info_in_main_log,
    save_metadata_about_this_analysis,
    save_roi_info_and_fit_outputs,
    save_signal_data_for_each_roi,
)


def test_get_sub_and_ses_case_no_log_file(
    one_folder_naming_specs, blueprint_spec
):
    sub, ses, reanalysis, analysis_log = get_sub_and_ses(
        one_folder_naming_specs, blueprint_spec
    )

    assert sub == 0
    assert ses == 0
    assert reanalysis is False
    assert isinstance(analysis_log, pd.DataFrame)


def test_get_sub_and_ses_case_analysis_log_empty(
    mocker, one_folder_naming_specs, blueprint_spec
):
    mocker.patch("pandas.read_csv", return_value=pd.DataFrame())
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)

    sub, ses, reanalysis, analysis_log = get_sub_and_ses(
        one_folder_naming_specs, blueprint_spec
    )

    assert sub == 0
    assert ses == 0
    assert reanalysis is False
    assert isinstance(analysis_log, pd.DataFrame)

    mock_file.assert_called_once_with(
        blueprint_spec.path / "analysis_log.csv", "r"
    )
    pd.read_csv.assert_called_once_with(mock_file(), index_col=0, header=0)


def test_get_sub_and_ses_case_subject_never_analysed_before(
    mocker, one_folder_naming_specs, blueprint_spec
):
    analysis_log = pd.DataFrame(
        {
            "folder name": ["folder_1", "folder_2", "folder_3"],
            "mouse line": ["line_1", "line_2", "line_3"],
            "mouse id": ["id_1", "id_2", "id_3"],
            "sub": [0, 1, 2],
            "ses": [0, 0, 0],
        }
    )
    mocker.patch("pandas.read_csv", return_value=analysis_log)
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)

    sub, ses, reanalysis, analysis_log = get_sub_and_ses(
        one_folder_naming_specs, blueprint_spec
    )

    assert sub == 3
    assert ses == 0
    assert reanalysis is False
    assert isinstance(analysis_log, pd.DataFrame)

    mock_file.assert_called_once_with(
        blueprint_spec.path / "analysis_log.csv", "r"
    )
    pd.read_csv.assert_called_once_with(mock_file(), index_col=0, header=0)


def test_get_sub_and_ses_case_subject_already_analysed(
    mocker, one_folder_naming_specs, blueprint_spec
):
    analysis_log = pd.DataFrame(
        {
            "folder_name": ["folder_1", "folder_2", "folder_3"],
            "mouse_line": [
                "line_1",
                "line_2",
                one_folder_naming_specs.mouse_line,
            ],
            "mouse_id": ["id_1", "id_2", one_folder_naming_specs.mouse_id],
            "sub": [0, 1, 2],
            "ses": [0, 0, 0],
        }
    )
    mocker.patch("pandas.read_csv", return_value=analysis_log)
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)

    sub, ses, reanalysis, analysis_log = get_sub_and_ses(
        one_folder_naming_specs, blueprint_spec
    )

    assert sub == 2
    assert ses == 1
    assert reanalysis is False
    assert isinstance(analysis_log, pd.DataFrame)

    mock_file.assert_called_once_with(
        blueprint_spec.path / "analysis_log.csv", "r"
    )
    pd.read_csv.assert_called_once_with(mock_file(), index_col=0, header=0)


def test_get_sub_and_ses_case_subject_and_session_already_analysed(
    mocker, one_folder_naming_specs, blueprint_spec
):
    analysis_log = pd.DataFrame(
        {
            "folder_name": [
                "folder_1",
                "folder_2",
                one_folder_naming_specs.folder_name,
            ],
            "mouse_line": [
                "line_1",
                "line_2",
                one_folder_naming_specs.mouse_line,
            ],
            "mouse_id": ["id_1", "id_2", one_folder_naming_specs.mouse_id],
            "sub": [0, 1, 2],
            "ses": [0, 0, 0],
        }
    )
    mocker.patch("pandas.read_csv", return_value=analysis_log)
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)

    sub, ses, reanalysis, analysis_log = get_sub_and_ses(
        one_folder_naming_specs, blueprint_spec
    )

    assert sub == 2
    assert ses == 0
    assert reanalysis is True
    assert isinstance(analysis_log, pd.DataFrame)


def test_save_roi_info_and_fit_outputs(photon_data, session_folder):
    Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

    save_roi_info_and_fit_outputs(photon_data, session_folder)

    assert (
        session_folder.ses_folder_path / "gaussians_fits_and_roi_info.pickle"
    ).exists()


def test_save_signal_data_for_each_roi(photon_data, session_folder):
    Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

    save_signal_data_for_each_roi(photon_data, session_folder)

    for roi in range(photon_data.n_roi):
        assert (
            session_folder.ses_folder_path
            / f"roi_{roi}_signal_dataframe.pickle"
        ).exists()


def test_save_metadata_about_this_analysis(
    photon_data,
    session_folder,
    one_folder_naming_specs,
    config,
):
    Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

    save_metadata_about_this_analysis(
        photon_data, session_folder, one_folder_naming_specs, config
    )

    assert (session_folder.ses_folder_path / "metadata.yml").exists()


def test_save_analysis_log(
    one_folder_naming_specs,
    subject_folder,
    session_folder,
    photon_data,
    blueprint_spec,
):
    Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

    reanalysis = False
    analysis_log = pd.DataFrame(
        {
            "folder_name": ["folder_1", "folder_2", "folder_3"],
            "mouse_line": ["line_1", "line_2", "line_3"],
            "mouse_id": ["id_1", "id_2", "id_3"],
            "sub": [0, 1, 2],
            "ses": [0, 0, 0],
        }
    )

    save_info_in_main_log(
        one_folder_naming_specs,
        subject_folder,
        session_folder,
        photon_data,
        reanalysis,
        analysis_log,
        blueprint_spec,
    )

    assert (blueprint_spec.path / "analysis_log.csv").exists()

    #  check if correct data is in the log
    analysis_log = pd.read_csv(
        blueprint_spec.path / "analysis_log.csv", index_col=0, header=0
    )
    assert analysis_log.shape == (4, 17)
    assert analysis_log[
        analysis_log["folder_name"] == one_folder_naming_specs.folder_name
    ].shape == (1, 17)
