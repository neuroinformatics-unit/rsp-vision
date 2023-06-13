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
    sub, ses, reanalysis, analysis_log = get_sub_and_ses(
        folder_naming_specs, swc_blueprint_spec
    )

    subject_folder = SubjectFolder(
        swc_blueprint_spec=swc_blueprint_spec,
        folder_or_table=folder_naming_specs,
        sub_num=sub,
    )

    session_folder = SessionFolder(
        subject_folder=subject_folder,
        folder_or_table=folder_naming_specs,
        ses_num=ses,
    )

    Path(session_folder.ses_folder_path).mkdir(parents=True, exist_ok=True)

    save_roi_info_and_fit_outputs(
        photon_data=photon_data,
        session_folder=session_folder,
    )

    save_signal_data_for_each_roi(
        photon_data=photon_data,
        session_folder=session_folder,
    )

    save_metadata_about_this_analysis(
        folder_naming_specs=folder_naming_specs,
        photon_data=photon_data,
        session_folder=session_folder,
        config_file=config_file,
    )

    save_info_in_main_log(
        folder_naming_specs=folder_naming_specs,
        photon_data=photon_data,
        session_folder=session_folder,
        analysis_log=analysis_log,
        subject_folder=subject_folder,
        reanalysis=reanalysis,
        swc_blueprint_spec=swc_blueprint_spec,
    )


def get_sub_and_ses(
    folder_naming_specs: FolderNamingSpecs,
    swc_blueprint_spec: SWC_Blueprint_Spec,
):
    reanalysis = False
    sub = None
    ses = None
    try:
        with open(swc_blueprint_spec.path / "analysis_log.csv", "r") as f:
            analysis_log = pd.read_csv(f, index_col=0, header=0)
            if analysis_log.empty:
                # CASE 1: no data at all
                sub = 0
                ses = 0
            elif analysis_log[
                (analysis_log["mouse line"] == folder_naming_specs.mouse_line)
                & (analysis_log["mouse id"] == folder_naming_specs.mouse_id)
            ].empty:
                # CASE 2: this subject was never analysed before
                sub = analysis_log["sub"].max() + 1
                ses = 0
            elif analysis_log[
                analysis_log["folder name"] == folder_naming_specs.folder_name
            ].empty:
                #  CASE 3: this subject was analysed before, but not this
                #  session folder_name is the specific dataset name,
                #  containing both info about the specific mouse and the
                #  specific session this is why we use it to determine if
                #  this session was analysed before
                this_line_rows = analysis_log[
                    (
                        analysis_log["mouse line"]
                        == folder_naming_specs.mouse_line
                    )
                    & (
                        analysis_log["mouse id"]
                        == folder_naming_specs.mouse_id
                    )
                ]
                assert len(this_line_rows) >= 1
                assert len(this_line_rows["sub"].unique()) == 1
                sub = this_line_rows["sub"].unique()[0]

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
                # CASE 4: this session was analysed before
                this_line_rows = analysis_log[
                    (
                        analysis_log["mouse line"]
                        == folder_naming_specs.mouse_line
                    )
                    & (
                        analysis_log["mouse id"]
                        == folder_naming_specs.mouse_id
                    )
                ]
                assert len(this_line_rows) >= 1
                assert len(this_line_rows["sub"].unique()) == 1
                sub = this_line_rows["sub"].unique()[0]

                ses = this_line_rows["ses"].max()
                reanalysis = True

    except FileNotFoundError:
        #  CASE 0: no analysis log file
        sub = 0
        ses = 0
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
    assert sub is not None
    assert ses is not None

    return sub, ses, reanalysis, analysis_log


def save_roi_info_and_fit_outputs(
    photon_data: PhotonData,
    session_folder: SessionFolder,
):
    # dict with subset of the data
    subset = {
        "n_roi": photon_data.n_roi,
        "responsive_rois": photon_data.responsive_rois,
        "median_subtracted_responses": photon_data.median_subtracted_response,
        "fit_outputs": photon_data.fit_output,
    }

    #  save part of the data in a pickle object
    with open(
        session_folder.ses_folder_path / "gaussians_fits_and_roi_info.pickle",
        "wb",
    ) as f:
        pickle.dump(subset, f)


def save_signal_data_for_each_roi(
    photon_data: PhotonData,
    session_folder: SessionFolder,
):
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


def save_metadata_about_this_analysis(
    photon_data: PhotonData,
    session_folder: SessionFolder,
    folder_naming_specs: FolderNamingSpecs,
    config_file: dict,
):
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


def save_info_in_main_log(
    folder_naming_specs: FolderNamingSpecs,
    subject_folder: SubjectFolder,
    session_folder: SessionFolder,
    photon_data: PhotonData,
    reanalysis: bool,
    analysis_log: pd.DataFrame,
    swc_blueprint_spec: SWC_Blueprint_Spec,
):
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
