import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rsp_vision.analysis.gaussians_calculations import (
    get_gaussian_matrix_to_be_plotted,
)
from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs
from rsp_vision.objects.SWC_Blueprint import (
    SessionFolder,
    SubjectFolder,
    SWC_Blueprint_Spec,
)

local_path = Path("/Users/lauraporta/local_data/rsp_vision/derivatives/")

merged_dataset = pd.DataFrame(
    columns=[
        "dataset_name",
        "roi_id",
        "total_rois_in_dataset",
        "penk",
        "is_responsive",
        "fit_correlation",
        "pval_fit_correlation",
        "preferred_sf",
        "preferred_tf",
        "peak_response",
        "exponential_factor",
        "sigma_sf",
        "sigma_tf",
        "peak_response_not_fitted",
    ]
)


all_non_penk = [
    "CX_1117646_hR_RSPd_cre-off_monitor_front",
    "CX_1117646_hL_RSPd_cre-off_monitor_front",
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


remote_path = Path("/Volumes/margrie/laura/")


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


for datagroup, group_name in zip(
    [all_penk, all_non_penk], ["penk", "non_penk"]
):
    paths = get_ROIs_paths(datagroup)
    is_penk = 1 if group_name == "penk" else 0
    for path, dataset in zip(paths, datagroup):
        print(f"{dataset}")
        with open(path / "gaussians_fits_and_roi_info.pickle", "rb") as f:
            data = pickle.load(f)

        n_rois = data["n_neurons"]
        neurons_idx = data["idx_neurons"]

        for roi_id in neurons_idx:
            msr = data["median_subtracted_responses"][(roi_id, "pooled")]
            peak_not_fitted = np.max(msr)
            fit_params = data["fit_outputs"][(roi_id, "pooled")]

            if np.isnan(np.min(fit_params)):
                print(f"ROI {roi_id} has NaN fit parameters")
                continue

            (
                peak_response,
                sf_0,
                tf_0,
                sigma_sf,
                sigma_tf,
                𝜻_power_law_exp,
            ) = fit_params
            gaussian = get_gaussian_matrix_to_be_plotted(
                kind="6x6 matrix",
                roi_id=roi_id,
                fit_output=data["fit_outputs"],
                direction="pooled",
                sfs=np.asarray([0.01, 0.02, 0.04, 0.08, 0.16, 0.32]),
                tfs=np.asarray([0.5, 1, 2, 4, 8, 16]),
                is_log=False,
            )
            correlation, pval = pearsonr(msr.flatten(), gaussian.flatten())
            df = pd.DataFrame(
                {
                    "dataset_name": dataset,
                    "roi_id": roi_id,
                    "total_rois_in_dataset": n_rois,
                    "penk": is_penk,
                    "is_responsive": 1
                    if roi_id in data["responsive_neurons"]
                    else 0,
                    "fit_correlation": correlation,
                    "pval_fit_correlation": pval,
                    "preferred_sf": sf_0,
                    "preferred_tf": tf_0,
                    "peak_response": peak_response,
                    "exponential_factor": 𝜻_power_law_exp,
                    "sigma_sf": sigma_sf,
                    "sigma_tf": sigma_tf,
                    "peak_response_not_fitted": peak_not_fitted,
                },
                index=[0],
            )

            merged_dataset = pd.concat([merged_dataset, df], ignore_index=True)


merged_dataset.to_csv(local_path / "merged_dataset_penk.csv")
