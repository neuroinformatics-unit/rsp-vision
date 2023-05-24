import datetime
import pickle
from pathlib import Path

import git
import pandas as pd
import yaml

from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs
from rsp_vision.objects.photon_data import PhotonData
from rsp_vision.objects.SWC_Blueprint import (
    SessionFolder,
    SubjectFolder,
    SWC_Blueprint_Spec,
)


def save_data(
    swc_blueprint_spec: SWC_Blueprint_Spec,
    folder_naming_specs: FolderNamingSpecs,
    photon_data: PhotonData,
    config_file: dict,
) -> None:
    # does the table exist?
    reanalysis = False
    try:
        with open(swc_blueprint_spec.path / "analysis_log.csv", "r") as f:
            analysis_log = pd.read_csv(f, index_col=0, header=0)
            if analysis_log.empty:
                # no data at all
                sub = 0
                ses = 0
            elif analysis_log[
                (analysis_log["mouse line"] == folder_naming_specs.mouse_line)
                & (analysis_log["mouse id"] == folder_naming_specs.mouse_id)
            ].empty:
                # this subject was never analysed before
                sub = analysis_log["sub"].max() + 1
                ses = 0
            elif analysis_log[
                analysis_log["folder name"] == folder_naming_specs.folder_name
            ].empty:
                #  this subject was analysed before, but not this session
                sub = analysis_log[
                    (
                        analysis_log["mouse line"]
                        == folder_naming_specs.mouse_line
                    )
                    & (
                        analysis_log["mouse id"]
                        == folder_naming_specs.mouse_id
                    )
                ]["sub"][0]
                ses = (
                    analysis_log[
                        (
                            analysis_log["mouse line"]
                            == folder_naming_specs.mouse_line
                        )
                        & (
                            analysis_log["mouse id"]
                            == folder_naming_specs.mouse_id
                        )
                    ]["ses"].max()
                    + 1
                )
            else:
                # this subject and this session were analysed before
                sub = analysis_log[
                    analysis_log["folder name"]
                    == folder_naming_specs.folder_name
                ]["sub"][0]
                ses = analysis_log[
                    analysis_log["folder name"]
                    == folder_naming_specs.folder_name
                ]["ses"][0]
                reanalysis = True

    except FileNotFoundError:
        analysis_log = pd.DataFrame(
            columns=[
                "folder name",
                "sub",
                "ses",
                "mouse line",
                "mouse id",
                "hemisphere",
                "brain region",
                "monitor position",
                "fov",
                "cre",
                "analysed",
                "analysis date",
                "commit hash",
                "microscope",
                "n roi",
                "n responsive roi",
                "days of the experiment",
            ],
        )

    subject_folder = SubjectFolder(
        swc_blueprint_spec=swc_blueprint_spec,
    ).make_from_folder_naming_specs(folder_naming_specs, sub)

    session_folder = SessionFolder(
        subject_folder=subject_folder,
    ).make_from_folder_naming_specs(folder_naming_specs, ses)

    Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

    # dict with subset of the data
    subset = {
        "n_roi": photon_data.n_roi,
        "responsive_rois": photon_data.responsive_rois,
        "downsampled_gaussians": photon_data.downsampled_gaussian,
        "oversampled_gaussians": photon_data.oversampled_gaussian,
        "median_subtracted_responses": photon_data.median_subtracted_response,
        "fit_outputs": photon_data.fit_output,
    }

    #  save part of the data in a pickle object
    with open(
        session_folder.ses_folder_path / "gaussians_fits_and_roi_info.pickle",
        "wb",
    ) as f:
        pickle.dump(subset, f)

    #  for each roi, save a subset of the signal dataframe in a pickle object
    for roi in range(photon_data.n_roi):
        with open(
            session_folder.ses_folder_path
            / f"roi_{roi}_signal_dataframe.pickle",
            "wb",
        ) as f:
            pickle.dump(
                photon_data.signal[photon_data.signal.roi_id == roi], f
            )

    #  save in a file details about the experimental data that was analysed
    metadata = {
        "folder_name": folder_naming_specs.folder_name,
        "photon_type": "two_photon"
        if photon_data.photon_type == PhotonType.TWO_PHOTON
        else "three_photon",
        "date": str(datetime.datetime.now()),
        "rsp_vision_latest_commit_hash": str(
            git.Repo(search_parent_directories=True).head.object.hexsha
        ),
        "config_file": config_file,
    }

    with open(session_folder.ses_folder_path / "metadata.yml", "w") as f:
        yaml.dump(metadata, f)

    dict = {
        "folder name": folder_naming_specs.folder_name,
        "sub": subject_folder.sub_num,
        "ses": session_folder.ses_num,
        "mouse line": folder_naming_specs.mouse_line,
        "mouse id": folder_naming_specs.mouse_id,
        "hemisphere": folder_naming_specs.hemisphere,
        "brain region": folder_naming_specs.brain_region,
        "monitor position": session_folder.monitor,
        "fov": folder_naming_specs.fov if folder_naming_specs.fov else "",
        "cre": folder_naming_specs.cre if folder_naming_specs.cre else "",
        "analysed": True,
        "analysis date": str(datetime.datetime.now()),
        "commit hash": str(
            git.Repo(search_parent_directories=True).head.object.hexsha
        ),
        "microscope": "two photon"
        if photon_data.photon_type == PhotonType.TWO_PHOTON
        else "three photon",
        "n roi": photon_data.n_roi,
        "n responsive roi": len(photon_data.responsive_rois),
        "days of the experiment": photon_data.total_n_days,
    }

    if not reanalysis:
        analysis_log = pd.concat([analysis_log, pd.DataFrame(dict, index=[0])])
    else:
        analysis_log.loc[
            analysis_log["folder name"] == folder_naming_specs.folder_name
        ] = dict

    analysis_log.to_csv(swc_blueprint_spec.path / "analysis_log.csv")
