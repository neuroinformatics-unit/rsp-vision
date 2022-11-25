import sys

import rich


def get_module_for_logging() -> object:
    """Get the name of the module for logging purposes.

    :return: name of the module
    :rtype: str
    """
    return sys.modules[__name__.partition(".")[0]]


def exception_handler(func):
    """Decorator to handle exceptions in the main function."""

    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            rich.print("Something went wrong 😱")
            rich.print(e)

    return inner_function
