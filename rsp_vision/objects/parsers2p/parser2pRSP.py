import logging
from pathlib import Path

from .parser2p import Parser2p


class Parser2pRSP(Parser2p):
    """Parses the folder name and evaluates the parameters `mouse_line`,
    `mouse_id`, `hemisphere`, `brain_region`, `monitor_position, `fov` and
    `cre`.

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
        super().__init__(folder_name, config)

    def _parse(self) -> dict:
        """Parses the folder name and evaluates the parameters
        `mouse_line`, `mouse_id`, `hemisphere`, `brain_region`,
        `monitor_position, `fov` and `cre`.

        Returns
        -------
        dict
            Dictionary containing the information parsed from the folder name

        Raises
        ------
        NotImplementedError
            if there is not the implementation for
            a specific folder structure
        RuntimeError
            if the parser failed in identifying the
            parameter `monitor_position`
        """
        info = {}

        splitted = self._folder_name.split("_")
        info["mouse_line"] = splitted[0]
        self._subfolder_exists = False

        if splitted[1].startswith("111"):
            info["mouse_id"] = splitted[1]
            info["hemisphere"] = splitted[2]
            info["brain_region"] = splitted[3]

        else:
            info["mouse_id"] = splitted[1] + "_" + splitted[2]

            if (info["mouse_line"] == "AS" and info["mouse_id"] == "95_2") or (
                info["mouse_line"] == "CX" and int(splitted[1]) < 61
            ):
                self._subfolder_exists = True
                logging.debug(
                    f"Experimental data without parsing implementation: \
                    {self._folder_name}"
                )
                raise NotImplementedError(
                    "Unclear data structure, contains subfolder"
                )

            info["hemisphere"] = splitted[3]
            info["brain_region"] = splitted[4]

        for item in splitted:
            if "FOV" in item:
                info["fov"] = item
            elif "cre" in item:
                info["cre"] = item
            elif "monitor" == item:
                info["monitor_position"] = item
            if "monitor_position" in info and item != "monitor":
                info["monitor_position"] += "_" + item

        if "monitor" not in info["monitor_position"]:
            logging.debug(
                "Monitor position not found in folder name",
                extra={"Parser2pRSP": self},
            )
            logging.debug(info["monitor_position"])
            raise RuntimeError("Monitor position not found in folder name")

        self.info = info

        return info

    def _get_parent_folder_name(self) -> str:
        """Returns the name of the parent folder which combines the name of
        the mouse line and the mouse id.

        Returns
        -------
        str
            name of the parent folder
        """
        return f'{self.info["mouse_line"]}_{self.info["mouse_id"]}'

    def get_path_to_experimental_folder(self) -> Path:
        """Returns the path to the folder containing the experimental data.

        Reads the server location from the config file and appends the parent
        folder and the given folder name.
        """
        return (
            Path(self._config["paths"]["imaging"])
            / Path(self._get_parent_folder_name())
            / Path(self._folder_name)
        )

    def get_path_to_allen_dff_file(self) -> Path:
        """Returns the path to the folder containing the allen dff files.

        Reads the server location from the config file and appends the parent
        folder and the given folder name.
        """
        filename = self._folder_name + "_sf_tf_allen_dff.mat"

        return Path(self._config["paths"]["allen-dff"]) / Path(filename)

    def get_path_to_serial2p(self) -> Path:
        """Returns the path to the folder containing the serial2p files.

        Reads the server location from the config file and appends the parent
        folder and the given folder name.
        """
        return Path(self._config["paths"]["serial2p"]) / Path(
            "CT_" + self._get_parent_folder_name()
        )

    def get_path_to_stimulus_analog_input_schedule_files(self) -> Path:
        """Returns the path to the folder containing the stimulus AI
        schedule files.

        Reads the server location from the config file and appends the parent
        folder and the given folder name.
        """
        return Path(self._config["paths"]["stimulus-ai-schedule"]) / Path(
            self._folder_name
        )
