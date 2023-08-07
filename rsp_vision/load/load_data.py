import logging
from pathlib import Path
from typing import Tuple

import h5py
import yaml

from rsp_vision.load.load_raw_suite2p_data import read_numpy_output_of_suite2p
from rsp_vision.load.load_stimulus_info import (
    check_how_many_sessions_in_dataset,
    how_many_days_in_dataset,
)

from ..objects.data_raw import DataRaw
from ..objects.enums import AnalysisType, DataType
from ..objects.folder_naming_specs import FolderNamingSpecs


def load_data(
    folder_name: str, config: dict
) -> Tuple[DataRaw, FolderNamingSpecs]:
    """Creates the configuration object and loads the data.

    Parameters
    ----------
    folder_name : string
        name of the folder containing the experimental data

    Returns
    -------
    Tuple(list, dict)
        specs: specs object
        data_raw: list containing all raw data
    """
    folder_naming = FolderNamingSpecs(folder_name, config)
    folder_naming.extract_all_file_names()
    data_raw = load_data_from_filename(folder_naming, config)

    return data_raw, folder_naming


def load_data_from_filename(
    folder_naming: FolderNamingSpecs, config: dict
) -> DataRaw:
    if config["use-allen-dff"]:
        if config["analysis-type"] == "sf_tf":
            allen_data_files = [
                file
                for file in folder_naming.all_files
                if file.datatype == DataType.ALLEN_DFF
                and file.analysistype == AnalysisType.SF_TF
            ]
            if len(allen_data_files) == 1:
                with h5py.File(allen_data_files[0].path, "r") as h5py_file:
                    data_raw = DataRaw(h5py_file, is_summary_data=True)

                logging.info("Summary data loaded")
                return data_raw
            else:
                raise ValueError(
                    "There is more than one summary file for sf_tf analysis"
                )
        else:
            raise NotImplementedError(
                "Only sf_tf analysis is implemented for summary data"
            )
    else:
        n_days = how_many_days_in_dataset(folder_naming)
        check_how_many_sessions_in_dataset(folder_naming, n_days)
        read_numpy_output_of_suite2p(folder_naming, n_days)


def read_config_file(config_path: Path) -> dict:
    """Reads the configuration file and returns the content as a
    dictionary.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file
    Returns
    -------
    dict
        content of the configuration file
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
