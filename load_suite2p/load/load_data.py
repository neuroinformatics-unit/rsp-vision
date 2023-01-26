import logging
from pathlib import Path
from typing import Tuple

import h5py
from decouple import config

from ..objects.data_raw import DataRaw
from ..objects.enums import AnalysisType, DataType
from ..objects.specifications import Specifications
from .read_config import read

CONFIG_PATH = config("CONFIG_PATH")
config_path = Path(__file__).parents[1] / CONFIG_PATH


def load_data(folder_name: str) -> Tuple[DataRaw, Specifications]:
    """Creates the configuration object and loads the data.

    Parameters
    ----------
    folder_name : string
        name of the folder containing the experimental data

    Returns
    -------
    Tuple(list, Specifications)
        specs: specs object
        data_raw: list containing all raw data
    """

    specs = get_specifications(folder_name)
    data_raw = load(specs)

    return data_raw, specs


def get_specifications(folder_name: str) -> Specifications:
    """Create specifications object. It reads the configuration
    file and adds the folder name and experimental details
    derived from it (analysis options and file paths).

    Parameters
    ----------
    folder_name : string
        name of the folder containing the experimental data

    Returns
    -------
    specs
        Specifications object
    """
    """"""
    logging.debug("Reading configurations")
    config = read(config_path)
    logging.debug(f"Configurations read: {config}")
    specs = Specifications(config, folder_name)
    return specs


def load(specs: Specifications) -> DataRaw:
    if specs.config["use-allen-dff"]:
        if specs.config["analysis-type"] == "sf_tf":
            allen_data_files = [
                file
                for file in specs.folder_naming.all_files
                if file.datatype == DataType.ALLEN_DFF
                and file.analysistype == AnalysisType.SF_TF
            ]
            if len(allen_data_files) == 1:
                with h5py.File(allen_data_files[0].path, "r") as h5py_file:
                    data_raw = DataRaw(h5py_file, is_allen=True)

                logging.info(f"Allen data loaded: {data_raw}")
                return data_raw
            else:
                raise ValueError(
                    "There is more than one Allen file for sf_tf analysis"
                )
        else:
            raise NotImplementedError(
                "Only sf_tf analysis is implemented for Allen data"
            )
    else:
        raise NotImplementedError("Only loading for Allen data is implemented")
