import pandas as pd
from rsp_vision.save.save_data import get_sub_and_ses

def test_get_sub_and_ses_case_no_log_file(folder_naming_specs, blueprint_spec):
    fns = folder_naming_specs[0]
    sub, ses, reanalysis, analysis_log = get_sub_and_ses(fns, blueprint_spec)

    assert sub == 0
    assert ses == 0
    assert reanalysis == False
    assert isinstance(analysis_log, pd.DataFrame)



def test_get_sub_and_ses_case_analysis_log_empty(mocker, folder_naming_specs, blueprint_spec):
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame())
    mock_file = mocker.mock_open()
    mocker.patch('builtins.open', mock_file)

    fns = folder_naming_specs[0]
    sub, ses, reanalysis, analysis_log = get_sub_and_ses(fns, blueprint_spec)

    assert sub == 0
    assert ses == 0
    assert reanalysis == False
    assert isinstance(analysis_log, pd.DataFrame)

    mock_file.assert_called_once_with(blueprint_spec.path / "analysis_log.csv", "r")
    pd.read_csv.assert_called_once_with(mock_file(), index_col=0, header=0)


def test_get_sub_and_ses_case_subject_never_analysed_before(mocker, folder_naming_specs, blueprint_spec):
    analysis_log = pd.DataFrame(
        {
            "folder name": ["folder_1", "folder_2", "folder_3"],
            "mouse line": ["line_1", "line_2", "line_3"],
            "mouse id": ["id_1", "id_2", "id_3"],
            "sub": [0, 1, 2],
            "ses": [0, 0, 0],
        }
    )
    mocker.patch('pandas.read_csv', return_value=analysis_log)
    mock_file = mocker.mock_open()
    mocker.patch('builtins.open', mock_file)

    fns = folder_naming_specs[0]
    sub, ses, reanalysis, analysis_log = get_sub_and_ses(fns, blueprint_spec)

    assert sub == 3
    assert ses == 0
    assert reanalysis == False
    assert isinstance(analysis_log, pd.DataFrame)

    mock_file.assert_called_once_with(blueprint_spec.path / "analysis_log.csv", "r")
    pd.read_csv.assert_called_once_with(mock_file(), index_col=0, header=0)

def test_get_sub_and_ses_case_subject_already_analysed(mocker, folder_naming_specs, blueprint_spec):
    fns = folder_naming_specs[0]

    analysis_log = pd.DataFrame(
        {
            "folder name": ["folder_1", "folder_2", "folder_3"],
            "mouse line": ["line_1", "line_2", fns.mouse_line],
            "mouse id": ["id_1", "id_2", fns.mouse_id],
            "sub": [0, 1, 2],
            "ses": [0, 0, 0],
        }
    )
    mocker.patch('pandas.read_csv', return_value=analysis_log)
    mock_file = mocker.mock_open()
    mocker.patch('builtins.open', mock_file)

    sub, ses, reanalysis, analysis_log = get_sub_and_ses(fns, blueprint_spec)

    assert sub == 2
    assert ses == 1
    assert reanalysis == False
    assert isinstance(analysis_log, pd.DataFrame)

    mock_file.assert_called_once_with(blueprint_spec.path / "analysis_log.csv", "r")
    pd.read_csv.assert_called_once_with(mock_file(), index_col=0, header=0)

def test_get_sub_and_ses_case_subject_and_session_already_analysed(mocker, folder_naming_specs, blueprint_spec):
    fns = folder_naming_specs[0]

    analysis_log = pd.DataFrame(
        {
            "folder name": ["folder_1", "folder_2", fns.folder_name],
            "mouse line": ["line_1", "line_2", fns.mouse_line],
            "mouse id": ["id_1", "id_2", fns.mouse_id],
            "sub": [0, 1, 2],
            "ses": [0, 0, 0],
        }
    )
    mocker.patch('pandas.read_csv', return_value=analysis_log)
    mock_file = mocker.mock_open()
    mocker.patch('builtins.open', mock_file)

    sub, ses, reanalysis, analysis_log = get_sub_and_ses(fns, blueprint_spec)

    assert sub == 2
    assert ses == 0
    assert reanalysis == True
    assert isinstance(analysis_log, pd.DataFrame)
