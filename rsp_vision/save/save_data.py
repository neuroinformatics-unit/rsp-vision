import datetime
import pickle
from pathlib import Path

import git
import pandas as pd
import yaml

from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs
from rsp_vision.objects.photon_data import PhotonData


class SWC_Blueprint_Spec:
    def __init__(
        self,
        project_name: str,
        raw_data: bool = False,
        derivatives: bool = True,
        local_path: Path = Path(),
    ) -> None:
        self.project_name = project_name
        self.raw_data = raw_data
        self.derivatives = derivatives
        self.path = (
            local_path / self.project_name / "derivatives"
            if derivatives
            else local_path / self.project_name / "raw_data"
        )
        self.logs_path = local_path / self.project_name / "logs"


class SubjectFolder:
    def __init__(
        self,
        swc_blueprint_spec: SWC_Blueprint_Spec,
        folder_naming_specs: FolderNamingSpecs,
    ) -> None:
        self.sub_num = self.get_latest_sub_number(swc_blueprint_spec)
        self.sub: str = f"sub-{self.sub_num:03d}"
        self.id = (
            "line-"
            + folder_naming_specs.mouse_line
            + "_id-"
            + folder_naming_specs.mouse_id
        )
        self.sub_folder_name = f"{self.sub}_{self.id}"
        self.sub_folder_path = Path(
            swc_blueprint_spec.path / self.sub_folder_name
        )

    def get_latest_sub_number(self, swc_blueprint_spec) -> int:
        try:
            onlyfolders = [
                f
                for f in swc_blueprint_spec.path.iterdir()
                if f.is_dir() and f.name.startswith("sub-")
            ]
            return int(onlyfolders[-1].name.split("_")[0][4:7])
        except FileNotFoundError:
            return 0


class SessionFolder:
    def __init__(
        self,
        subject_folder: SubjectFolder,
        folder_naming_specs: FolderNamingSpecs,
    ) -> None:
        self.ses_num = self.get_latest_ses_number(subject_folder)
        self.ses = f"ses-{self.ses_num:03d}"
        self.monitor = (
            "_".join(folder_naming_specs.monitor_position.split("_")[1:])
            .replace("_", "-")
            .replace("-", "")
        )
        self.id = (
            "hemisphere-"
            + folder_naming_specs.hemisphere
            + "_region-"
            + folder_naming_specs.brain_region
            + "_monitor-"
            + self.monitor
            + (
                "_fov-" + folder_naming_specs.fov
                if folder_naming_specs.fov
                else ""
            )
            + (
                "_cre-" + folder_naming_specs.cre
                if folder_naming_specs.cre
                else ""
            )
        )
        self.ses_folder_name = f"{self.ses}_{self.id}"
        self.ses_folder_path = Path(
            subject_folder.sub_folder_path / self.ses_folder_name
        )

    def get_latest_ses_number(self, subject_folder) -> int:
        try:
            onlyfolders = [
                f
                for f in subject_folder.sub_folder_path.iterdir()
                if f.is_dir() and f.name.startswith("ses-")
            ]
            return int(onlyfolders[-1].name.split("_")[0][4:7])
        except FileNotFoundError:
            return 0


def save_data(
    swc_blueprint_spec: SWC_Blueprint_Spec,
    folder_naming_specs: FolderNamingSpecs,
    photon_data: PhotonData,
    config_file: dict,
) -> None:
    subject_folder = SubjectFolder(
        swc_blueprint_spec=swc_blueprint_spec,
        folder_naming_specs=folder_naming_specs,
    )

    session_folder = SessionFolder(
        subject_folder=subject_folder,
        folder_naming_specs=folder_naming_specs,
    )

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

    #  read csv containing analysis log
    with open(swc_blueprint_spec.path / "analysis_log.csv", "r") as f:
        analysis_log = pd.read_csv(f, index_col=0, header=0)
        #  try to see if there is a row with the same folder name
        #  if there is, update the row
        #  if there is not, append a new row
        #  save the csv
        dict = {
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
        if analysis_log[
            (analysis_log["sub"] == subject_folder.sub_num)
            & (analysis_log["ses"] == session_folder.ses_num)
        ].empty:
            analysis_log = analysis_log.append(
                dict,
                ignore_index=True,
            )
        else:
            analysis_log.loc[
                (analysis_log["sub"] == subject_folder.sub_num)
                & (analysis_log["ses"] == session_folder.ses_num),
                dict.keys(),
            ] = dict.values()

        analysis_log.to_csv(swc_blueprint_spec.path / "analysis_log.csv")
