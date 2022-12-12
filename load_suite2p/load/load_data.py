import logging
from pathlib import Path
from typing import Tuple

from read_config import read

from ..objects.configurations import Config

config_path = Path(__file__).parent / "config/config.yml"


def load_data(folder_name: str) -> Tuple[Config, list]:
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

    config = configure(folder_name)
    data_raw = load(config)

    return config, data_raw


def configure(folder_name: str) -> Config:
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

    config = Config(read_configurations(), folder_name)
    return config


def load(config: Config) -> list:
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
