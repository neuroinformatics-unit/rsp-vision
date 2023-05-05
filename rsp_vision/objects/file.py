from pathlib import Path

from .enums import AnalysisType, DataType


class File:
    """Class containing the name of the file, its path and its extension.

    Attributes
    ----------
    name: str
        file name

    path: Path
        complete file path

    extension: str
        file extension
    """

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path: Path = path
        self._path_str = str(path)
        self.datatype: DataType = self._get_data_type()
        self.analysistype: AnalysisType = self._get_analysis_type()

    def _get_data_type(self) -> DataType:
        if (
            "suite2p" in self._path_str
            or "plane0" in self._path_str
            or "Fall.mat" in self._path_str
        ):
            return DataType.SIGNAL
        elif "stimulus_info.mat" in self._path_str:
            return DataType.STIMULUS_INFO
        elif "trigger_info.mat" in self._path_str:
            return DataType.TRIGGER_INFO
        elif "rocro_reg.mat" in self._path_str:
            return DataType.REGISTERS2P
        elif "allen_dff.mat" in self._path_str:
            return DataType.ALLEN_DFF
        else:
            return DataType.NOT_FOUND

    def _get_analysis_type(self) -> AnalysisType:
        if "sf_tf" in str(self._path_str):
            return AnalysisType.SF_TF
        elif "sparse_noise" in str(self._path_str):
            return AnalysisType.SPARSE_NOISE
        elif "retinotopy" in str(self._path_str):
            return AnalysisType.RETINOTOPY
        else:
            return AnalysisType.UNCLASSIFIED
