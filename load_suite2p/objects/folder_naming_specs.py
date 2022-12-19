import logging
from pathlib import Path

from .parsers2p.parser2pRSP import Parser2pRSP


class FolderNamingSpecs:
    """The class :class:`FolderNamingSpecs` represents the naming convention
    of the files and folders in which the experimental data is stored.

    Attributes
    ----------
    folder_name : str
        Name of the folder containing the experimental
        data generated by suite2p and registers2p

    config: dict
        Dictionary containing the configuration parameters

    mouse_line : str
        Name of the mouse line (e.g. 'CX')

    mouse_id : str
        Mouse id (e.g. '95_2')

    hemisphere : str
        Hemisphere of the brain (e.g. 'hL')

    brain_region : str
        Brain region (e.g. 'V1')

    monitor_position : str
        Monitor position (e.g. 'monitor_front')

    fov : str, optional
        Field of view (e.g. 'fov1')

    cre : str, optional
        Cre line (e.g. 'Sst-IRES-Cre')
    """

    def __init__(
        self,
        folder_name: str,
        config: dict,
    ):
        self.original_config = config

        self.folder_name = folder_name

        logging.info("Parsing folder name")
        self.parse_name()
        self.mouse_line = self._parser.info["mouse_line"]
        self.mouse_id = self._parser.info["mouse_id"]
        self.hemisphere = self._parser.info["hemisphere"]
        self.brain_region = self._parser.info["brain_region"]
        self.monitor_position = self._parser.info["monitor_position"]

        try:
            self.fov = self._parser.info["fov"]
        except KeyError:
            self.fov = None

        try:
            self.cre = self._parser.info["cre"]
        except KeyError:
            self.cre = None

        if not self.check_if_folder_exists():
            logging.error(f"File {self.get_path()} does not exist")
            raise FileNotFoundError(
                f"File {self.folder_name} not found. "
                + "Please check the file name and try again."
            )

    def parse_name(self) -> None:
        """Parses the folder name and evaluates the parameters `mouse_line`,
        `mouse_id`, `hemisphere`, `brain_region`, `monitor_position.
        Other parameters might be parsed depending on the project.

        Raises
        ------
        ValueError
            if the parser specified in the config file is
        not implemented
        """

        if self.original_config["parser"] == "Parser2pRSP":
            logging.debug("Parsing folder name using Parser2pRSP")
            self._parser = Parser2pRSP(self.folder_name, self.original_config)
        else:
            logging.error(
                f"Parser {self.original_config['parser']} \
                not supported"
            )
            raise ValueError(
                f"Parser {self.original_config['parser']} \
                not supported"
            )

    def get_path(self) -> Path:
        """Returns the path to the folder containing the experimental data.
        Reads the server location from the config file and appends the
        parent folder and the given folder name.

        Returns
        -------
        Path
            path to the folder containing the experimental data
        """
        return self._parser.get_path()

    def check_if_folder_exists(self) -> bool:
        """Checks if the folder containing the experimental data exists.
        The folder path is obtained by calling the method :meth:`get_path`.

        Returns
        -------
        bool
            True if folder exists, False otherwise
        """
        return self.get_path().exists()

    def extract_all_file_names(self) -> list:
        # get filenames by day
        # search for files called 'suite2p', 'plane0', 'Fall.mat'
        # get session names to get name of stim files
        # corrects for exceptions
        raise NotImplementedError("This method is not implemented yet")
