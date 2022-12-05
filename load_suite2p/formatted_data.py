from .folder_naming_specs import FolderNamingSpecs


class FormattedData:
    """Class to load the formatted data from suite2p and registers2p."""

    def __init__(self, file_name: str, config: dict):
        self.file_name = file_name
        self.file_specs = self.get_FolderNamingSpecs(config)

    def get_FolderNamingSpecs(self, config) -> FolderNamingSpecs:
        return FolderNamingSpecs(self.file_name, config)
