import numpy as np

from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs


def read_numpy_output_of_suite2p(
    folder_naming_specs: FolderNamingSpecs, n_days: int
):
    data = {}
    for day in range(1, n_days + 1):
        path_to_plane0 = get_plane0_folder_path(folder_naming_specs, day)

        ops = np.load(path_to_plane0 / "ops.npy", allow_pickle=True)
        stat = np.load(path_to_plane0 / "stat.npy", allow_pickle=True)
        F = np.load(path_to_plane0 / "F.npy", allow_pickle=True)
        Fneu = np.load(path_to_plane0 / "Fneu.npy", allow_pickle=True)
        iscell = np.load(path_to_plane0 / "iscell.npy", allow_pickle=True)
        spks = np.load(path_to_plane0 / "spks.npy", allow_pickle=True)

        data[day] = {
            "ops": ops,
            "stat": stat,
            "F": F,
            "Fneu": Fneu,
            "iscell": iscell,
            "spks": spks,
        }

    return data


def get_suite2p_folder_path(
    folder_naming_specs: FolderNamingSpecs, day_number: int
):
    return (
        folder_naming_specs.path_to_experiment_folder
        / f"day_{day_number}"
        / "suite2p"
    )


def get_plane0_folder_path(
    folder_naming_specs: FolderNamingSpecs, day_number: int
):
    return get_suite2p_folder_path(folder_naming_specs, day_number) / "plane0"
