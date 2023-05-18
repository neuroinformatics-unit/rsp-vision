import logging
import sys
from pathlib import Path

import rich
from decouple import config
from fancylog import fancylog
from rich.prompt import Prompt

from rsp_vision.analysis.spatial_freq_temporal_freq import (
    FrequencyResponsiveness,
)
from rsp_vision.load.load_data import load_data, read_config_file
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData
from rsp_vision.save.save_data import SWC_Blueprint_Spec, save_data

CONFIG_PATH = config("CONFIG_PATH")
config_path = Path(__file__).parents[1] / CONFIG_PATH


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
    config = read_config_file(config_path)
    swc_blueprint_spec = SWC_Blueprint_Spec(
        project_name="rsp_vision",
        raw_data=False,
        derivatives=True,
        local_path=Path(config["paths"]["output"]),
    )
    start_logging(swc_blueprint_spec)
    logging.debug(f"Config file read from {config_path}")
    logging.debug(f"Config file content: {config}")

    folder_name = Prompt.ask(
        " \
        Please provide the experimental folder name.\n \
        Format: Mouse_Id_Hemisphere_BrainRegion_Monitor_position.\n \
        Example (1 day): AK_1111739_hL_RSPd_monitor_front\n \
        Example (2 days): BY_IAA_1117276_hR_RSPg_monitor_front\n \
        Example (1 day, BIG): AS_1112809_hL_V1_monitor_front-right_low\n \
        Example (2 days, big file): CX_1112654_hL_RSPd_monitor_front\n \
        Example (2 days, big file): CX_1112837_hL_RSPd_monitor_front\n \
        📁"
    )

    # load data
    data, folder_naming = load_data(folder_name, config)

    # preprocess and make PhotonData object
    photon_data = PhotonData(data, PhotonType.TWO_PHOTON, config)

    # make analysis object
    responsiveness = FrequencyResponsiveness(photon_data)

    # calculate responsiveness and save results in PhotonData object
    photon_data = responsiveness()

    logging.info("Analysis finished")
    logging.info(f"Updated photon_data object: {photon_data}")

    # save results
    save_data(swc_blueprint_spec, folder_naming, photon_data, config)


def start_logging(swc_blueprint_spec: SWC_Blueprint_Spec, module=None):
    """Start logging to file and console.

    The log level to file is set to DEBUG, to console to INFO. The log file is
    saved in the current working directory. Uses fancylog to format the log
    messages.
    """
    if module is None:
        module = get_module_for_logging()

    fancylog.start_logging(
        output_dir=str(swc_blueprint_spec.logs_path),
        package=module,
        filename="rsp_vision",
        verbose=False,
    )


def get_module_for_logging() -> object:
    """Get the name of the module for logging purposes.

    Returns
    -------
    object
        name of the module
    """
    return sys.modules[__name__.partition(".")[0]]
