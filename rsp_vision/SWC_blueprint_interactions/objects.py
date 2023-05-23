from pathlib import Path

from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs


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
    def __init__(self, swc_blueprint_spec: SWC_Blueprint_Spec):
        self.swc_blueprint_spec = swc_blueprint_spec

    def make_from_folder_naming_specs(
        self, folder_naming_specs: FolderNamingSpecs
    ):
        self.sub_num = self.get_latest_sub_number(self.swc_blueprint_spec)
        self.sub = f"sub-{self.sub_num:03d}"
        self.id = (
            "line-"
            + folder_naming_specs.mouse_line
            + "_id-"
            + folder_naming_specs.mouse_id
        )
        self.sub_folder_name = f"{self.sub}_{self.id}"
        self.sub_folder_path = Path(
            self.swc_blueprint_spec.path / self.sub_folder_name
        )
        return self

    def make_from_table_row(self, table_row: dict):
        self.sub_num = int(table_row["sub"])
        self.sub = f"sub-{self.sub_num:03d}"
        self.id = (
            "line-"
            + table_row["mouse line"]
            + "_id-"
            + str(table_row["mouse id"])
        )
        self.sub_folder_name = f"{self.sub}_{self.id}"
        self.sub_folder_path = Path(
            self.swc_blueprint_spec.path / self.sub_folder_name
        )
        return self

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
    ):
        self.subject_folder = subject_folder

    def make_from_folder_naming_specs(
        self, folder_naming_specs: FolderNamingSpecs
    ):
        self.ses_num = self.get_latest_ses_number(self.subject_folder)
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
            self.subject_folder.sub_folder_path / self.ses_folder_name
        )
        return self

    def make_from_table_row(self, table_row: dict):
        self.ses_num = int(table_row["ses"])
        self.ses = f"ses-{self.ses_num:03d}"
        self.monitor = table_row["monitor position"]
        self.id = (
            "hemisphere-"
            + table_row["hemisphere"]
            + "_region-"
            + table_row["brain region"]
            + "_monitor-"
            + self.monitor
            + (
                "_fov-" + str(table_row["fov"])
                if (str(table_row["fov"]) != "nan")
                else ""
            )
            + (
                "_cre-" + str(table_row["cre"])
                if (str(table_row["cre"]) != "nan")
                else ""
            )
        )

        self.ses_folder_name = f"{self.ses}_{self.id}"
        self.ses_folder_path = Path(
            self.subject_folder.sub_folder_path / self.ses_folder_name
        )
        return self

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
