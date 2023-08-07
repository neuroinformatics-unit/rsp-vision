import os

from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs


def load_analog_input_files(folder_naming_specs: FolderNamingSpecs):
    # load AI.bin
    # load AI_info.mat
    # load stimulus_info.mat
    pass


def how_many_days_in_dataset(folder_naming_specs: FolderNamingSpecs) -> int:
    # count number of folders in main dataset folder
    n_days = len(os.listdir(folder_naming_specs.path_to_experiment_folder))

    assert n_days in [1, 2, 3], "There should be 1, 2 or 3 days in the dataset"

    return n_days


def check_how_many_sessions_in_dataset(
    folder_naming_specs: FolderNamingSpecs, n_days: int
) -> None:
    # count number of folders in main dataset folder for each day.
    # should be 18, this is practically a sanity check

    path = folder_naming_specs.path_to_experiment_folder
    day_folders = [f"day_{i}" for i in range(1, n_days + 1)]
    for day_folder in day_folders:
        # one folder is the suite2p output folder
        assert (
            len(os.listdir(path / day_folder)) == 19
        ), f"There should be 18 sessions in {day_folder}"


def load_trigger_info(folder_naming_specs: FolderNamingSpecs):
    # load trigger_info.mat
    pass
