import logging
import pickle
import sys

import rich
from fancylog import fancylog
from rich.prompt import Prompt

from rsp_vision.analysis.spatial_freq_temporal_freq import (
    FrequencyResponsiveness,
)
from rsp_vision.load.load_data import load_data
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData


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
            logging.exception(e)

    return inner_function


@exception_handler
def analysis_pipeline() -> None:
    """Entry point of the program. CLI or GUI functionality is added here."""
    # pipeline draft
    start_logging()

    # TODO: add TUI or GUI fuctionality to get input from user
    folder_name = Prompt.ask(
        " \
        Please provide the experimental folder name.\n \
        Format: Mouse_Id_Hemisphere_BrainRegion_Monitor_position.\n \
        Example: AK_1111739_hL_RSPd_monitor_front\n \
        📁"
    )

    # load data
    data, config = load_data(folder_name)

    # preprocess and make PhotonData object
    photon_data = PhotonData(data, PhotonType.TWO_PHOTON, config)

    # make analysis object
    responsiveness = FrequencyResponsiveness(photon_data)

    # calculate responsiveness and save results in PhotonData object
    photon_data = responsiveness()

    logging.info("Analysis finished")
    logging.info(f"Updated photon_data object: {photon_data}")

    with open(f"{folder_name}_data.pickle", "wb") as f:
        pickle.dump(photon_data, f)
        logging.info("Analysis saved")


def start_logging(module=None):
    """Start logging to file and console. The log level to file is set to
    DEBUG, to console to INFO. The log file is saved in the current working
    directory. Uses fancylog to format the log messages.
    """
    if module is None:
        module = get_module_for_logging()

    fancylog.start_logging(
        output_dir="./", package=module, filename="load_suite2p", verbose=False
    )


def get_module_for_logging() -> object:
    """Get the name of the module for logging purposes.

    Returns
    -------
    object
        name of the module
    """
    return sys.modules[__name__.partition(".")[0]]