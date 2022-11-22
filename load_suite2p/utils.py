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

    :return: True if the machine is connected to the VPN, False otherwise
    """
    config = read()
    return (
        True if int(os.system("ping -c 1 " + config["server"])) == 0 else False
    )


def is_winstor_mounted() -> bool:
    """Checks if the winstor folder is mounted.

    :return: True if the winstor folder is mounted, False otherwise
    """
    config = read()
    return True if os.path.ismount(config["paths"]["winstor"]) else False


def exception_handler(func):
    """Decorator to handle exceptions in the main function."""

    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            rich.print("Something went wrong 😱")
            rich.print(e)

    return inner_function
