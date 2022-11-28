import logging

import rich
from fancylog import fancylog
from rich.prompt import Prompt
from vpn_server_connections.connections import (
    can_ping_swc_server,
    is_winstor_mounted,
)

from .load.folder_naming_specs import FolderNamingSpecs
from .read_config import read
from .utils import exception_handler, get_module_for_logging


def start_logging():
    """Start logging to file and console. The log level to file is set to
    DEBUG, to console to INFO. The log file is saved in the current working
    directory. Uses fancylog to format the log messages.
    """
    module = get_module_for_logging()

    fancylog.start_logging(
        output_dir="./", package=module, filename="load_suite2p", verbose=False
    )


def read_configurations():
    """Read configurations regarding experiment and analysis.

    :return: dictionary with configurations
    :rtype: dict
    """

    logging.debug("Reading configurations")
    config = read()
    logging.debug(f"Configurations read: {config}")

    return config


def check_connection(config: dict) -> None:
    """Check if the connection to the server is established and if Winstor
    is mounted.
    """

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

    config = read()
    check_connection(config)
    folder_name = Prompt.ask("Please provide the folder name")
    file_naming_specs = FolderNamingSpecs(folder_name, config)

    path = file_naming_specs.get_path()
    rich.print("Success! 🎉")
    rich.print(path)
