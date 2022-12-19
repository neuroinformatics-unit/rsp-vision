from .folder_naming_specs import FolderNamingSpecs
from .options import Options


class Specifications:
    """The class :class:`Specifications` holds the configuration of the
    experiment, the analysis options, and the paths to the files
    to be loaded."""

    def __init__(self, config: dict, folder_name: str):
        self.base_paths: dict = config["paths"]
        self.folder_name = folder_name
        self.folder_naming_specs = FolderNamingSpecs(folder_name, config)
        self.folder_naming_specs.extract_all_file_names()
        self.options = Options(config)
