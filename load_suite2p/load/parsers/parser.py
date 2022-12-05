from abc import ABC, abstractmethod

from path import Path


class Parser(ABC):
    """Abstract base class for parsers. Child classes must be project
    specific and tailor made to the folder structure in Winstor.
    When inheriting from this class, the method parse() must be implemented.

    Attributes
    ----------

    info : dict
        Dictionary containing the information parsed from the folder name
        It must contain the following keys:
        - mouse_line
        - mouse_id
        - hemisphere
        - brain_region
        - monitor_position
        It may contain the following
        - fov
        - cre

    Raises
    ------
    ValueError
        if there is not the minimum number of
        parameters set by the parser in the child
        class.
    """

    def __init__(self, folder_name: str, config: dict) -> None:

        self._folder_name = folder_name
        self.info = self._parse()
        if not self._minimum_params_required():
            raise ValueError(
                "The minimum parameters required are not present. "
                + "Please check the parser implementation."
            )
        self._config = config

    @abstractmethod
    def _parse(self) -> dict:
        """Parses the folder name and evaluates the parameters `mouse_line`,
        `mouse_id`, `hemisphere`, `brain_region` and `monitor_position`.
        To be implemented by the child classes taking into account
        the folder structure of each project.
        """
        pass

    @abstractmethod
    def get_path(self) -> Path:
        """Returns the path to the file containing the suite2p output. To be
        implemented by the child classes taking into account the folder
        structure of each project.
        """
        pass

    def _minimum_params_required(self) -> bool:
        """Checks if the minimum parameters have been evaluated by the parser.

        Returns
        -------
        bool
            True if the minimum parameters are present, False otherwise
        """

        mandatory_keys = [
            "mouse_line",
            "mouse_id",
            "hemisphere",
            "brain_region",
            "monitor_position",
        ]

        for key in mandatory_keys:
            if key not in self.info.keys():
                return False

        return True
