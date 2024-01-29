import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs
from rsp_vision.objects.SWC_Blueprint import (
    SessionFolder,
    SubjectFolder,
    SWC_Blueprint_Spec,
)

remote_path = Path("/Volumes/margrie-1/laura/")
local_plots_path = Path(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/"
)


all_non_penk = [
    "CX_1117646_hR_RSPd_cre-off_monitor_front",
    "SG_1118210_hR_RSPd_cre-off_monitor_front",
    "SG_1117788_hR_RSPd_cre-off_monitor_front",
    "CX_1117217_hR_RSPd_cre-off_monitor_front",
]

all_penk = [
    "CX_142_3_hL_RSPd_monitor_front",
    "CX_142_3_hR_RSPd_monitor_front",
    "CX_122_2_hR_RSPd_monitor_right",
    "CX_102_2_hL_RSPd_FOV1_monitor_right",
    "CX_102_2_hL_RSPd_FOV3_monitor_right",
]

swc_blueprint_spec = SWC_Blueprint_Spec(
    project_name="rsp_vision",
    raw_data=False,
    derivatives=True,
    local_path=remote_path,
)

config = {
    "parser": "Parser2pRSP",
    "use-allen-dff": True,
    "paths": {
        "imaging": "/path/to/",
        "allen-dff": "/path/to/allen_dff",
        "serial2p": "/path/to/serial2p",
        "stimulus-ai-schedule": "/path/to/stimulus_AI_schedule_files",
        "output": "/path/to/output",
    },
}

log_table = pd.read_csv(swc_blueprint_spec.path / "analysis_log.csv")


def get_ROIs_paths(data_group):
    paths = []

    for dataset in data_group:
        names = FolderNamingSpecs(dataset, config)
        row = log_table[log_table["folder_name"] == dataset]
        sub_num = row["sub"].values[0]
        ses_num = row["ses"].values[0]
        subject = SubjectFolder(swc_blueprint_spec, names, sub_num)
        session = SessionFolder(subject, names, ses_num)

        paths.append(session.ses_folder_path)

    return paths


activity: Dict[str, Dict[str, Dict[str, list]]] = {}
for datagroup, name in zip([all_penk, all_non_penk], ["penk", "non_penk"]):
    paths = get_ROIs_paths(datagroup)

    activity[name] = {}
    for path, dataset in zip(paths, datagroup):
        print(f"{dataset}: ", end=" ")
        with open(path / "gaussians_fits_and_roi_info.pickle", "rb") as f:
            info = pickle.load(f)
        n_roi = info["n_roi"]

        baseline: List[np.ndarray] = []
        response: List[np.ndarray] = []
        for i in range(n_roi):
            print("*", end="")
            with open(path / f"roi_{i}_signal_dataframe.pickle", "rb") as f:
                df = pickle.load(f)

            print(df.shape)
