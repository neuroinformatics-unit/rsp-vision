import logging
import sys

import rich
from fancylog import fancylog
from vpn_server_connections.connections import (
    can_ping_swc_server,
    is_winstor_mounted,
)


def get_module_for_logging() -> object:
    """Get the name of the module for logging purposes.

    Returns
    -------
    object
        name of the module
    """
    return sys.modules[__name__.partition(".")[0]]


def exception_handler(func: object) -> object:
    """Decorator to handle exceptions in the main function.

    Parameters
    ----------
    func : object
        function to decorate

    Returns
    -------
    object
        decorated function
    """

    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            rich.print("Something went wrong 😱")
            rich.print(e)

    return inner_function


def start_logging():
    """Start logging to file and console. The log level to file is set to
    DEBUG, to console to INFO. The log file is saved in the current working
    directory. Uses fancylog to format the log messages.
    """
    module = get_module_for_logging()

    fancylog.start_logging(
        output_dir="./", package=module, filename="load_suite2p", verbose=False
    )


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
