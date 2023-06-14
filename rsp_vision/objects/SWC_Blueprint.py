from pathlib import Path
from typing import Union

from rsp_vision.objects.folder_naming_specs import FolderNamingSpecs


class SWC_Blueprint_Spec:
    def __init__(
        self,
        project_name: str,
        raw_data: bool = False,
        derivatives: bool = True,
        local_path: Path = Path("."),
    ) -> None:
        """
        Create a `SWC_Blueprint_Spec` object to specify the location of the
        raw data and/or derivatives for a project.
        Please refer to https://swc-blueprint.neuroinformatics.dev/ for more
        information about SWC Blueprint.

        Parameters
        ----------
        project_name : str
            The name of the project folder.
        raw_data : bool, optional
            Whether you want to create a subfolder for raw data or not,
            by default False.
        derivatives : bool, optional
            Whether you want to create a subfolder for derivatives or not,
            by default True.
        local_path : Path, optional
            The path to the local folder where you want to create the project,
            by default Path(".").
        """
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
    """
    A class to represent a subject folder in SWC Blueprint.
    This specific implementation is for the rsp_vision project.
    The subject folder will have the following format:
    sub-###_line-###_id-### (e.g. sub-001_line-C57_id-123).

    Line and id are specified in the `FolderNamingSpecs` object or
    in the table row. The table is a pandas DataFrame that contains
    the information about the subject that were already analysed.

    Attributes
    ----------
    swc_blueprint_spec : SWC_Blueprint_Spec
        The SWC Blueprint specification for the project.
    sub_num : int
        The subject number.
    sub_folder_name : str
        The name of the subject folder. It contains the subject number,
        the mouse line and the mouse id.
    sub_folder_path : Path
        The path to the subject folder.

    Methods
    -------
    make_from_folder_naming_specs(folder_naming_specs, sub_num)
        Create a `SubjectFolder` object from a `FolderNamingSpecs` object
        and a subject number.
    make_from_table_row(table_row)
        Create a `SubjectFolder` object from a table row.

    Raises
    ------
    ValueError
        If the argument `folder_or_table` is neither 'folder' nor 'table'.
    """

    def __init__(
        self,
        swc_blueprint_spec: SWC_Blueprint_Spec,
        folder_or_table: Union[FolderNamingSpecs, dict],
        sub_num: int = 0,
    ):
        self.swc_blueprint_spec = swc_blueprint_spec
        if isinstance(folder_or_table, FolderNamingSpecs):
            self.make_from_folder_naming_specs(folder_or_table, sub_num)
        elif isinstance(folder_or_table, dict):
            self.make_from_table_row(folder_or_table)
        else:
            raise ValueError(
                "The argument `folder_or_table` must be an instance of "
                + "`FolderNamingSpecs` or a dictionary."
            )
        self.sub_folder_path = Path(
            self.swc_blueprint_spec.path / self.sub_folder_name
        )

    def make_from_folder_naming_specs(
        self,
        folder_naming_specs: FolderNamingSpecs,
        sub_num: int,
    ):
        self.sub_num = sub_num
        self.sub_folder_name = (
            f"sub-{sub_num:03d}"
            + "_line-"
            + folder_naming_specs.mouse_line
            + "_id-"
            + folder_naming_specs.mouse_id
        )

    def make_from_table_row(self, table_row: dict):
        self.sub_num = int(table_row["sub"])
        self.sub_folder_name = (
            f"sub-{self.sub_num:03d}"
            + "_line-"
            + table_row["mouse line"]
            + "_id-"
            + str(table_row["mouse id"])
        )


class SessionFolder:
    """
    A class to represent a session folder in SWC Blueprint.
    This specific implementation is for the rsp_vision project.
    The session folder will have the following format:
    ses-###_hemisphere-###_region-###_monitor-###_fov-###_cre-### (e.g.
    ses-000_hemisphere-hL_region-RSPd_monitor-front);
    `fov` and `cre` are optional.

    Attributes
    ----------
    subject_folder : SubjectFolder
        The subject folder.
    ses_num : int
        The session number.
    monitor : str
        The monitor position.
    ses_folder_name : str
        The name of the session folder. It contains the session number,
        the hemisphere, the brain region, the monitor position, the field
        of view and the cre line (if applicable).
    ses_folder_path : Path
        The path to the session folder.

    Methods
    -------
    make_from_folder_naming_specs(folder_naming_specs, ses_num)
        Create a `SessionFolder` object from a `FolderNamingSpecs` object
        and a session number.

    make_from_table_row(table_row)
        Create a `SessionFolder` object from a table row.

    Raises
    ------
    ValueError
        If the argument `folder_or_table` is neither 'folder' nor 'table'.
    """

    def __init__(
        self,
        subject_folder: SubjectFolder,
        folder_or_table: Union[FolderNamingSpecs, dict],
        ses_num: int = 0,
    ):
        self.subject_folder = subject_folder
        if isinstance(folder_or_table, FolderNamingSpecs):
            self.make_from_folder_naming_specs(folder_or_table, ses_num)
        elif isinstance(folder_or_table, dict):
            self.make_from_table_row(folder_or_table)
        else:
            raise ValueError(
                "The argument `folder_or_table` must be an instance of "
                + "`FolderNamingSpecs` or a dictionary."
            )
        self.ses_folder_path = Path(
            self.subject_folder.sub_folder_path / self.ses_folder_name
        )

    def make_from_folder_naming_specs(
        self,
        folder_naming_specs: FolderNamingSpecs,
        ses_num: int,
    ):
        self.ses_num = ses_num
        self.monitor = (
            "_".join(folder_naming_specs.monitor_position.split("_")[1:])
            .replace("_", "-")
            .replace("-", "")
        )
        self.ses_folder_name = (
            f"ses-{self.ses_num:03d}"
            + "_hemisphere-"
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

    def make_from_table_row(self, table_row: dict):
        self.ses_num = int(table_row["ses"])
        self.monitor = table_row["monitor position"]
        self.ses_folder_name = (
            f"ses-{self.ses_num:03d}"
            + "_hemisphere-"
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
