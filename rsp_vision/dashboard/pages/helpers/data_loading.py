import pickle
from pathlib import Path


def load_data(store: dict) -> dict:
    """This method loads the data from the pickle file.

    Parameters
    ----------
    store : dict
        The store object.

    Returns
    -------
    dict
        The data from the pickle file.
    """
    path = (
        Path(store["path"])
        / store["subject_folder_path"]
        / store["session_folder_path"]
        / "gaussians_fits_and_roi_info.pickle"
    )
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def load_data_of_signal_dataframe(store: dict, roi_id: int) -> dict:
    """This method loads the data of single ROIs from the signal dataframe.

    Parameters
    ----------
    store : dict
        The store object containing the path to the data.
    roi_id : int
        The id of the ROI.

    Returns
    -------
    dict
        A subset of the signal dataframe containing the data of the ROI.
    """
    path = (
        Path(store["path"])
        / store["subject_folder_path"]
        / store["session_folder_path"]
    )

    with open(path / f"roi_{roi_id}_signal_dataframe.pickle", "rb") as f:
        signal = pickle.load(f)

    return signal
