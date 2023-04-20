import logging
import pickle

from rich.prompt import Prompt

from rsp_vision.analysis.spatial_freq_temporal_freq import FrequencyAnalysis
from rsp_vision.load.load_data import load_data
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData
from rsp_vision.utils import exception_handler, start_logging


@exception_handler
def main():
    """Entry point of the program. CLI or GUI functionality is added here."""
    # pipeline draft
    start_logging()

    # TODO: add TUI or GUI fuctionality to get input from user
    folder_name = Prompt.ask(
        " \
        Please provide the experimental folder name.\n \
        Format: Mouse_Id_Hemisphere_BrainRegion_Monitor_position.\n \
        Example: AK_1111739_hL_RSPd_monitor_front\n \
        📁"
    )

    photon_type = PhotonType.TWO_PHOTON
    # load data
    data, config = load_data(folder_name)

    # preprocess and make PhotonData object
    photon_data = PhotonData(data, PhotonType.TWO_PHOTON, config)

    # make analysis object
    analysis = FrequencyAnalysis(photon_data, photon_type)

    # calculate responsiveness and display it in a nice way
    analysis.responsiveness()

    with open(f"{folder_name}_analysis.pickle", "wb") as f:
        pickle.dump(analysis, f)
        logging.info("Analysis saved")
