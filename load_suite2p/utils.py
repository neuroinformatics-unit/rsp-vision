import sys


def get_module_for_logging() -> object:
    """Get the name of the module for logging purposes.

    :return: name of the module
    :rtype: str
    """
    return sys.modules[__name__.partition(".")[0]]
