from .folder_naming_specs import FolderNamingSpecs


class FormattedData:
    """Class to load the formatted data from suite2p and registers2p."""

    def __init__(
        self,
        file_name: str,
    ):
        self.file_name = file_name
        self.file_specs = self.get_FileNamingSpecs()

    def get_FileNamingSpecs(self) -> FolderNamingSpecs:
        return FolderNamingSpecs(self.file_name)
