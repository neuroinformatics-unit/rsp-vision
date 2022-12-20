from enum import Enum


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
