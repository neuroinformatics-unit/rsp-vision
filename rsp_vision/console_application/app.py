import datetime
import logging
import os
import sys
from pathlib import Path

from decouple import config
from fancylog import fancylog
from rich.prompt import Prompt

from rsp_vision.analysis.spatial_freq_temporal_freq import (
    FrequencyResponsiveness,
)
from rsp_vision.load.load_data import load_data, read_config_file
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData
from rsp_vision.objects.SWC_Blueprint import SWC_Blueprint_Spec
from rsp_vision.save.save_data import save_data
from rsp_vision.save.tables_manager import AnalysisSuccessTable

CONFIG_PATH = config("CONFIG_PATH")
config_path = Path(__file__).parents[1] / CONFIG_PATH


def cli_entry_point_local():
    """
    This is the entry point for the CLI application when running locally.
    It suggests a series of datasets to analyse and then runs the analysis.
    """
    folder_name = Prompt.ask(
        " \
        Please provide only the dataset name.\n \
        Format: Mouse_Id_Hemisphere_BrainRegion_Monitor_position.\n \
        Example (1 day): AK_1111739_hL_RSPd_monitor_front\n \
        Example (2 days): BY_IAA_1117276_hR_RSPg_monitor_front\n \
        Example (1 day, BIG): AS_1112809_hL_V1_monitor_front-right_low\n \
        Example (2 days, big file): CX_1112654_hL_RSPd_monitor_front\n \
        Example (2 days, big file): CX_1112837_hL_RSPd_monitor_front\n \
        Error: CX_122_2_hR_RSPd_monitor_right\n \
        📄"
    )
    config, swc_blueprint_spec = read_config_and_logging()
    analysis_pipeline(folder_name, config, swc_blueprint_spec)


def get_all_sf_tf_datasets(path: str) -> list:
    """Returns a list of all the datasets in the folder containing the string
    "sf_tf" in their name.

    Parameters
    ----------
    path : str
        Path to the folder containing the datasets.

    Returns
    -------
    list
        List of all the datasets in the folder containing the string "sf_tf"
        in their name.
    """

    only_sf_tf_files = []
    for filename in os.listdir(path):
        if "sf_tf" in filename:
            filename = filename.split("_sf_tf")[0]
            only_sf_tf_files.append(filename)
    only_sf_tf_files.sort()
    return only_sf_tf_files


def cli_entry_point_array(job_id):
    """This is the entry point for the CLI application when running on the
    cluster. It takes a job id as input and runs the analysis on the dataset
    corresponding to that job id. I.e. if the job id is 0, it will run the
    analysis on the first dataset in the folder containing the string "sf_tf"
    in its name.
    It also updates the analysis_success.log file with the results of the
    analysis. If there is an error, it will be logged in the error column of
    the analysis_success.log file.

    Parameters
    ----------
    job_id : _type_
        Job id of the dataset to analyse.
    """
    config, swc_blueprint_spec = read_config_and_logging(
        is_local=False, job_id=job_id
    )

    allen_folder = config["paths"]["allen-dff"]
    only_sf_tf_files = get_all_sf_tf_datasets(allen_folder)
    dataset = only_sf_tf_files[job_id]

    analysis_success_table = AnalysisSuccessTable(swc_blueprint_spec.path)

    logging.info(f"Trying to analyse:{dataset}, job id: {job_id}")
    try:
        row = analysis_success_table.find_this_dataset(dataset)
        if row["state"].values[0] == "Analysis successful 🥳":
            logging.info("Dataset already analysed")
        else:
            analysis_success_table.update(
                dataset_name=dataset,
                date=str(datetime.datetime.now()),
                latest_job_id=job_id,
                state="Starting the analysis...",
            )
            analysis_pipeline(dataset, config, swc_blueprint_spec)
            analysis_success_table.update(
                dataset_name=dataset,
                date=str(datetime.datetime.now()),
                latest_job_id=job_id,
                state="Analysis successful 🥳",
            )

    except Exception as e:
        error = str(e)
        logging.exception(e)
        analysis_success_table.update(
            dataset_name=dataset,
            date=str(datetime.datetime.now()),
            latest_job_id=job_id,
            state="⚠️ error: " + error,
        )


def read_config_and_logging(is_local=True, job_id=0) -> tuple:
    """Reads the config file and starts logging.

    Parameters
    ----------
    is_local : bool, optional
        Whether the analysis is run locally or on the cluster, by default True
    job_id : int, optional
        Job id of the dataset to analyse, by default 0

    Returns
    -------
    tuple
        Tuple containing the config and the swc_blueprint_spec.
    """
    config = read_config_file(config_path)
    swc_blueprint_spec = SWC_Blueprint_Spec(
        project_name="rsp_vision",
        raw_data=False,
        derivatives=True,
        local_path=Path(config["paths"]["output"])
        if is_local
        else Path(config["batch-paths"]["output"]),
    )
    if not is_local:
        config["paths"] = config["batch-paths"]
    start_logging(swc_blueprint_spec, job_id=job_id)
    logging.debug(f"Config file read from {config_path}")
    logging.debug(f"Config file content: {config}")

    return config, swc_blueprint_spec


# @exception_handler
def analysis_pipeline(folder_name, config, swc_blueprint_spec) -> None:
    """Current analysis pipeline. Loads the data, preprocesses it, runs the
    analysis and saves the results.

    Parameters
    ----------
    folder_name : _type_
        The name of the dataset to analyse.
    config : _type_
        The config file.
    swc_blueprint_spec : _type_
        The swc_blueprint_spec object.
    """
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


def start_logging(
    swc_blueprint_spec: SWC_Blueprint_Spec, module=None, job_id=0
):
    """Start logging to file and console.

    The log level to file is set to DEBUG, to console to INFO. The log file is
    saved in the current working directory. Uses fancylog to format the log
    messages.
    """
    if module is None:
        module = get_module_for_logging()

    Path(swc_blueprint_spec.logs_path).mkdir(parents=True, exist_ok=True)

    fancylog.start_logging(
        output_dir=str(swc_blueprint_spec.logs_path),
        package=module,
        filename=f"rsp_vision_{job_id}",
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
