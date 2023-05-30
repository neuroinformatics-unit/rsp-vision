import logging
import pickle
import sys
from pathlib import Path

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
    """Entry point of the program.

    CLI or GUI functionality is added here.
    """
    # pipeline draft
    start_logging()

    folder_name = Prompt.ask(
        " \
        Please provide only the dataset name.\n \
        Format: Mouse_Id_Hemisphere_BrainRegion_Monitor_position.\n \
        Example (1 day): AK_1111739_hL_RSPd_monitor_front\n \
        Example (2 days): BY_IAA_1117276_hR_RSPg_monitor_front\n \
        Example (1 day, BIG): AS_1112809_hL_V1_monitor_front-right_low\n \
        Example (2 days, big file): CX_1112654_hL_RSPd_monitor_front\n \
        Example (2 days, big file): CX_1112837_hL_RSPd_monitor_front\n \
        📄"
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

    saving_path = (
        Path(config["paths"]["output"]) / f"{folder_name}_data.pickle"
    )
    with open(saving_path, "wb") as f:
        pickle.dump(photon_data, f)
        logging.info("Analysis saved")


def start_logging(module=None):
    """Start logging to file and console.

    The log level to file is set to DEBUG, to console to INFO. The log file is
    saved in the current working directory. Uses fancylog to format the log
    messages.
    """
    if module is None:
        module = get_module_for_logging()

    fancylog.start_logging(
        output_dir="./", package=module, filename="rsp_vision", verbose=False
    )


def get_module_for_logging() -> object:
    """Get the name of the module for logging purposes.

    Returns
    -------
    object
        name of the module
    """
    return sys.modules[__name__.partition(".")[0]]
