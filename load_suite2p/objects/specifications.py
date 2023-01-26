from .folder_naming_specs import FolderNamingSpecs
from .options import Options


class Specifications:
    """The class :class:`Specifications` holds the configuration of the
    experiment, the analysis options, and the paths to the files
    to be loaded."""

    def __init__(self, config: dict, folder_name: str):
        self.config: dict = config
        self.folder_name = folder_name
        self.folder_naming = FolderNamingSpecs(folder_name, config)
        self.folder_naming.extract_all_file_names()
        self.options = Options(config)
