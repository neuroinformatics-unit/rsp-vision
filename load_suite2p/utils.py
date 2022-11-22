import os
import sys

import rich

from .read_config import read


def get_module_for_logging() -> object:
    """Get the name of the module for logging purposes.

    :return: name of the module
    :rtype: str
    """
    return sys.modules[__name__.partition(".")[0]]


def can_ping_swc_server() -> bool:
    """Checks if the machine is connected to the VPN.

    :return: True if Winstor server is connected, False otherwise
    """
    config = read()
    return True if os.system("ping -c 1 " + config["server"]) == 0 else False


def exception_handler(func):
    """Decorator to handle exceptions in the main function."""

    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            rich.print("Something went wrong 😱")
            rich.print(e)

    return inner_function
