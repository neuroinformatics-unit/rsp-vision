import datetime
import pickle
from pathlib import Path

import git
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


class SubjectFolder:
    def __init__(
        self,
        swc_blueprint_spec: SWC_Blueprint_Spec,
        folder_naming_specs: FolderNamingSpecs,
    ) -> None:
        self.sub: str = (
            f"sub-{self.get_latest_sub_number(swc_blueprint_spec):03d}"
        )
        self.id = (
            "id-"
            + folder_naming_specs.mouse_line
            + "-"
            + folder_naming_specs.mouse_id
        ).replace("_", "-")
        self.sub_folder_name = f"{self.sub}_{self.id}"
        self.sub_folder_path = Path(
            swc_blueprint_spec.path / self.sub_folder_name
        )

    def get_latest_sub_number(self, swc_blueprint_spec) -> int:
        # get the list of folders in the project folder
        # get the last sub folder
        # if does not exist, return 0

        return 1


class SessionFolder:
    def __init__(
        self,
        subject_folder: SubjectFolder,
        folder_naming_specs: FolderNamingSpecs,
    ) -> None:
        self.ses = f"ses-{self.get_latest_ses_number(subject_folder):03d}"
        self.id = (
            "id-"
            + folder_naming_specs.folder_name.replace("_", "-")[
                len(subject_folder.id) :
            ]
        )
        self.ses_folder_name = f"{self.ses}_{self.id}"
        self.ses_folder_path = Path(
            subject_folder.sub_folder_path / self.ses_folder_name
        )

    def get_latest_ses_number(self, subject_folder) -> int:
        # get the list of folders in the subject folder
        # get the last ses folder
        # if does not exist, return 0

        return 1


def save_data(
    folder_naming_specs: FolderNamingSpecs,
    photon_data: PhotonData,
    config_file: dict,
) -> None:
    swc_blueprint_spec = SWC_Blueprint_Spec(
        project_name="rsp_vision",
        raw_data=False,
        derivatives=True,
        local_path=Path("/Users/lauraporta/local_data/"),
    )

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
