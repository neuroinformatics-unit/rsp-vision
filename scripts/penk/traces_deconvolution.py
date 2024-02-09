import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from oasis.functions import deconvolve
from tqdm import tqdm

from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs
from rsp_vision.objects.SWC_Blueprint import (
    SessionFolder,
    SubjectFolder,
    SWC_Blueprint_Spec,
)

remote_path = Path("/ceph/margrie/laura/")


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
        n_roi = info["n_neurons"]

        deconvolved_traces = {}
        for i in tqdm(
            info["idx_neurons"],
            total=n_roi,
            desc=f"Processing {dataset}",
        ):
            with open(path / f"roi_{i}_signal_dataframe.pickle", "rb") as f:
                df = pickle.load(f)

            # take deltaF/F traces
            deltaF_overF = df.signal.values
            if len(np.where(np.isnan(deltaF_overF))[0]) > 0:
                cutoff = np.where(np.isnan(deltaF_overF))[0][0]
            else:
                cutoff = len(deltaF_overF)
            deltaF_overF = deltaF_overF[:cutoff]

            # deconvolve them
            # we are not providing the baseline, so it will be estimated
            # c is the calcium concentration
            # s is predicted spiking activity
            try:
                c, s, b, g, lam = deconvolve(deltaF_overF, penalty=1)
            except np.linalg.LinAlgError as e:
                print(f"Error in {dataset} {i}: {e}")
                s = np.asarray([np.nan] * len(deltaF_overF))

            deconvolved_traces[i] = s

        activity[name][dataset] = deconvolved_traces


with open(swc_blueprint_spec.path / "deconvolved_traces.pickle", "wb") as g:
    pickle.dump(activity, g)
