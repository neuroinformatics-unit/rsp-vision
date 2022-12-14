from .folder_naming_specs import FolderNamingSpecs
from .options import Options


class Config:
    """The class :class:`Config` represents the configuration of the
    experiment and the analysis."""

    def __init__(self, config: dict, folder_name: str):
        self.base_paths: dict = config["paths"]
        self.folder_name = folder_name

        self.folder_naming_specs = FolderNamingSpecs(folder_name, config)
        self.all_filenames = self.folder_naming_specs.extract_all_file_names()
        (
            self.signal,
            self.stimulus_info,
            self.trigger_info,
            self.registers2p,
        ) = self.folder_naming_specs.categorize(self.all_filenames)

        self.options = Options(config)
