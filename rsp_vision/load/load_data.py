import logging
from pathlib import Path
from typing import Tuple

import h5py
from decouple import config

from ..load.read_config import read
from ..objects.data_raw import DataRaw
from ..objects.enums import AnalysisType, DataType
from ..objects.folder_naming_specs import FolderNamingSpecs

CONFIG_PATH = config("CONFIG_PATH")
config_path = Path(__file__).parents[1] / CONFIG_PATH


def load_data(folder_name: str) -> Tuple[DataRaw, dict]:
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

    config = read(config_path)
    folder_naming = FolderNamingSpecs(folder_name, config)
    folder_naming.extract_all_file_names()
    data_raw = load(folder_naming, config)

    return data_raw, config


def load(folder_naming: FolderNamingSpecs, config: dict) -> DataRaw:
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
                    data_raw = DataRaw(h5py_file, is_allen=True)

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
        raise NotImplementedError(
            "Only loading for summary data is implemented"
        )
