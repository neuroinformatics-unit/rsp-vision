import logging
from pathlib import Path

import yaml


def read(config_path: Path) -> dict:
    """Reads the configuration file and returns the content as a dictionary.

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
        logging.debug(f"Config file read from {config_path}")
        logging.debug(f"Config file content: {config}")
    return config