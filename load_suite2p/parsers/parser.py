from abc import ABC, abstractmethod


class Parser(ABC):
    """Abstract base class for parsers"""

    def __init__(self, folder_name: str) -> None:
        self.folder_name = folder_name
        self.parse()

    @abstractmethod
    def parse(self) -> None:
        pass
