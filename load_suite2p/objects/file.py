from enum import Enum
from pathlib import Path


class File:
    """Class containing the name of the file, its path and its
        extension.

    Attributes
    ----------
    name: str
        file name

    path: Path
        complete file path

    extension: str
        file extension
    """

    class DataType(Enum):
        SIGNAL = 1
        STIMULUS_INFO = 2
        TRIGGER_INFO = 3
        REGISTERS2P = 4
        ALLEN_DFF = 5

    class AnalysisType(Enum):
        SF_TF = 1
        SPARSE_NOISE = 2
        RETINOTOPY = 3
        UNCLASSIFIED = 4

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path: Path = path
        self._path_str = str(path)
        self.datatype = self._get_data_type()
        self.analysis_type = self._get_analysis_type()

    def _get_data_type(self) -> DataType:
        if (
            "suite2p" in self._path_str
            or "plane0" in self._path_str
            or "Fall.mat" in self._path_str
        ):
            return self.DataType.SIGNAL
        elif "stimulus_info.mat" in self._path_str:
            return self.DataType.STIMULUS_INFO
        elif "trigger_info.mat" in self._path_str:
            return self.DataType.TRIGGER_INFO
        elif "rocro_reg.mat" in self._path_str:
            return self.DataType.REGISTERS2P
        elif "allen_dff.mat" in self._path_str:
            return self.DataType.ALLEN_DFF
        else:
            raise ValueError("File not to be used")

    def _get_analysis_type(self) -> AnalysisType:
        if "sf_tf" in str(self._path_str):
            return self.AnalysisType.SF_TF
        elif "sparse_noise" in str(self._path_str):
            return self.AnalysisType.SPARSE_NOISE
        elif "retinotopy" in str(self._path_str):
            return self.AnalysisType.RETINOTOPY
        else:
            return self.AnalysisType.UNCLASSIFIED
