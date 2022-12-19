import logging
from pathlib import Path
from typing import Tuple

from ..objects.specifications import Specifications
from .read_config import read

config_path = Path(__file__).parents[1] / "config/config.yml"


def load_data(folder_name: str) -> Tuple[list, Specifications]:
    """Creates the configuration object and loads the data.

    Parameters
    ----------
    folder_name : string
        name of the folder containing the experimental data

    Returns
    -------
    Tuple(Config, list)
        config: configuration object
        data_raw: list containing all raw data
    """

    specs = get_specifications(folder_name)
    data_raw = load(specs)

    return data_raw, specs


def get_specifications(folder_name: str) -> Specifications:
    """Create configuration object. It reads the configuration
    file and adds the folder name and experimental details
    derived from it.

    Parameters
    ----------
    folder_name : string
        name of the folder containing the experimental data

    Returns
    -------
    Config
        Configuration object
    """
    """"""

    specs = Specifications(read_configurations(), folder_name)
    return specs


def load(specs: Specifications) -> list:
    raise NotImplementedError("TODO")


def read_configurations() -> dict:
    """Read configurations regarding experiment and analysis.

    Returns
    -------
    dict
        dictionary with configurations
    """

    logging.debug("Reading configurations")
    config = read(config_path)
    logging.debug(f"Configurations read: {config}")

    return config
