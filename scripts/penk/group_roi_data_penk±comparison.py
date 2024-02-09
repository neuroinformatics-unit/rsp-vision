import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rsp_vision.analysis.gaussians_calculations import (
    get_gaussian_matrix_to_be_plotted,
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

for is_penk, group in enumerate([all_non_penk, all_penk]):
    group_name = "all_non_penk" if is_penk == 0 else "all_penk"
    for dataset in group:
        print(f"Processing {dataset}")
        file_name = dataset + ".pickle"
        all_penk_one_dataset = local_path / group_name / file_name

        with open(all_penk_one_dataset, "rb") as f:
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
