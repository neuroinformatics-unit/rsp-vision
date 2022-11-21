import os

from path import Path

from .parsers.chryssanthi import ChryssanthiParser
from .read_config import read


class FormattedData:
    def __init__(self, raw_data, formatted_data):
        self.raw_data = raw_data
        self.formatted_data = formatted_data


class FileNamingSpecs:
    def __init__(
        self,
        folder_name: str,
    ):
        self.folder_name = folder_name
        self.config = read()
        self.parse_name()

        if not self.check_if_file_exists():
            # log error
            print(self.get_path())
            raise FileNotFoundError(
                f"File {self.folder_name} not found. "
                + "Please check the file name and try again."
            )

    def parse_name(self) -> None:
        if self.config["scientist"] == "Chryssanthi":
            # log
            parser = ChryssanthiParser(self.folder_name)

            self.mouse_line = parser.mouse_line
            self.mouse_id = parser.mouse_id
            self.hemisphere = parser.hemisphere
            self.brain_region = parser.brain_region
            self.monitor_position = parser.monitor_position

            if hasattr(parser, "fov"):
                self.fov = parser.fov
            if hasattr(parser, "cre"):
                self.cre = parser.cre

    def get_folder_name(self) -> str:
        return f"{self.mouse_line}_{self.mouse_id}"

    def get_path(self) -> Path:
        # could contain other subfolders
        return (
            Path(self.config["paths"]["imaging"])
            / Path(self.get_folder_name())
            / Path(self.folder_name)
        )

    def check_if_file_exists(self) -> bool:
        return os.path.exists(self.get_path())


class RawData:
    def __init__(
        self,
        file_name: str,
    ):
        self.file_name = file_name
        self.file_specs = self.get_FileNamingSpecs()

    def get_FileNamingSpecs(self) -> FileNamingSpecs:
        return FileNamingSpecs(self.file_name)
