import logging

from .parser import Parser


class ChryssanthiParser(Parser):
    """Parser for Chryssanthi's data"""

    def __init__(self, folder_name: str) -> None:
        super().__init__(folder_name)

    def parse(self) -> None:
        splitted = self.folder_name.split("_")
        self.mouse_line = splitted[0]
        self.subfolder_exists = False

        if splitted[1].startswith("111"):
            self.mouse_id = splitted[1]
            self.hemisphere = splitted[2]
            self.brain_region = splitted[3]

        else:
            self.mouse_id = splitted[1] + "_" + splitted[2]

            if (self.mouse_line == "AS" and self.mouse_id == "95_2") or (
                self.mouse_line == "CX" and int(splitted[1]) < 61
            ):
                self.subfolder_exists = True
                logging.warning(
                    f"Experimental data without parsing implementation: \
                    {self.folder_name}"
                )
                raise NotImplementedError(
                    "Unclear data structure, contains subfolder"
                )

            self.hemisphere = splitted[3]
            self.brain_region = splitted[4]

        for item in splitted:
            if "FOV" in item:
                self.fov = item
            if "cre" in item:
                self.cre = item
            if "monitor" == item:
                self.monitor_position = item
            if hasattr(self, "monitor_position"):
                self.monitor_position += item

        if "monitor" not in self.monitor_position:
            logging.error(
                "Monitor position not found in folder name",
                extra={"ChryssanthiParser": self},
            )
            logging.debug(self.monitor_position)
            raise RuntimeError("Monitor position not found in folder name")
