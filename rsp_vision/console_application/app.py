import datetime
import logging
import os
import sys
from pathlib import Path

import pandas as pd
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

CONFIG_PATH = config("CONFIG_PATH")
config_path = Path(__file__).parents[1] / CONFIG_PATH


def cli_entry_point_local():
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


def cli_entry_point_array(job_id):
    config, swc_blueprint_spec = read_config_and_logging(
        is_local=False, job_id=job_id
    )

    allen_folder = config["paths"]["allen-dff"]
    only_sf_tf_files = []
    for filename in os.listdir(allen_folder):
        if "sf_tf" in filename:
            filename = filename.split("_sf_tf")[0]
            only_sf_tf_files.append(filename)

    dataset = only_sf_tf_files[job_id]

    all_datasets_logging_file_path = (
        swc_blueprint_spec.local_path / "analysis_success.log"
    )
    reanalysis = False
    try:
        with open(all_datasets_logging_file_path, "r") as f:
            local_logs = pd.read_csv(f, index_col=0, header=0)
            if dataset in local_logs["dataset_name"]:
                reanalysis = True
    except Exception:
        local_logs = pd.DataFrame(
            columns=["dataset_name", "date", "latest_job_id"]
        )

    error = ""
    logging.info(f"Trying to analyse:{dataset}, job id: {job_id}")
    try:
        analysis_pipeline(dataset, config, swc_blueprint_spec)
    except Exception as e:
        error = str(e)
        logging.exception(e)

    ll = {
        "dataset_name": dataset,
        "date": str(datetime.datetime.now()),
        "latest_job_id": job_id,
        "error": error,
    }

    if not reanalysis:
        local_logs = pd.concat([local_logs, pd.DataFrame(ll, index=[0])])
    else:
        local_logs.loc[local_logs["dataset_name"] == dataset] == ll

    local_logs.to_csv(all_datasets_logging_file_path)


def read_config_and_logging(is_local=True, job_id=0):
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
