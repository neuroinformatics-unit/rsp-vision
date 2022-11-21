import logging

import rich
from fancylog import fancylog
from rich.prompt import Prompt

from .data_objects import FileNamingSpecs
from .read_config import read
from .utils import get_module_for_logging


def start_logging():
    module = get_module_for_logging()

    fancylog.start_logging(
        output_dir="./", package=module, filename="load_suite2p"
    )


def read_configurations():
    """Read configurations regarding experiment and analysis.

    :return: dictionary with configurations
    :rtype: dict
    """

    logging.info("Reading configurations")
    config = read()
    logging.info(f"Configurations read: {config}")

    return config


def main():
    start_logging()
    folder_name = Prompt.ask("Please provide the folder name")

    file_naming_specs = FileNamingSpecs(folder_name)
    path = file_naming_specs.get_path()
    rich.print(path)
