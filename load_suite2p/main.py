import logging
from pathlib import Path

import rich
from fancylog import fancylog
from load.folder_naming_specs import FolderNamingSpecs
from rich.prompt import Prompt
from vpn_server_connections.connections import (
    can_ping_swc_server,
    is_winstor_mounted,
)

from .read_config import read
from .utils import exception_handler, get_module_for_logging

config_path = Path(__file__).parent / "config/config.yml"


def start_logging():
    """Start logging to file and console. The log level to file is set to
    DEBUG, to console to INFO. The log file is saved in the current working
    directory. Uses fancylog to format the log messages.
    """
    module = get_module_for_logging()

    fancylog.start_logging(
        output_dir="./", package=module, filename="load_suite2p", verbose=False
    )


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


def check_connection(config: dict):
    """Check if the connection to the server is established and if Winstor
    is mounted.

    Parameters
    ----------
    config : dict
        Configuration file to load the data

    Raises
    ------
    RuntimeError
        If the connection to the VPN is not established
    RuntimeError
        If Winstor is not mounted
    RuntimeError
        If the configuration file is not correct
    """
    if not (
        ("server" in config)
        and ("paths" in config)
        and ("winstor" in config["paths"])
    ):
        raise RuntimeError(
            "The configuration file is not complete."
            + "Please check the documentation."
        )

    if not can_ping_swc_server(config["server"]):
        logging.debug("Please connect to the VPN.")
        raise RuntimeError("Please connect to the VPN.")
    if not is_winstor_mounted(config["paths"]["winstor"]):
        logging.debug("Please mount Winstor.")
        raise RuntimeError("Please mount Winstor.")


@exception_handler
def main():
    """Main function of the package. It starts logging, reads the
    configurations, asks the user to input the folder name and then
    instantiates a :class:`FolderNamingSpecs` object.
    """
    start_logging()

    config = read(config_path)
    check_connection(config)
    folder_name = Prompt.ask("Please provide the folder name")
    folder_naming_specs = FolderNamingSpecs(folder_name, config)

    path = folder_naming_specs.get_path()
    rich.print("Success! 🎉")
    rich.print(path)
