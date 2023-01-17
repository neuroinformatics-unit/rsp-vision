from rich.prompt import Prompt

from .analysis.spatial_freq_temporal_freq import SF_TF
from .load.load_data import load_data
from .objects.photon_data import PhotonData
from .plots.plotter import Plotter
from .utils import exception_handler, start_logging


@exception_handler
def main():
    """Entry point of the program. CLI or GUI functionality is added here."""
    # pipeline draft
    start_logging()

    # TODO: add TUI or GUI fuctionality to get input from user
    folder_name = Prompt.ask("Please provide the folder name")

    # load data
    data, specs = load_data(folder_name)

    # preprocess and make PhotonData object
    photon_data = PhotonData(data)

    # make analysis object
    analysis = SF_TF(photon_data, specs)

    # calculate responsiveness and display it in a nice way
    responsiveness = analysis.responsiveness(photon_data)
    print(responsiveness)  # TODO: nice appearance

    # Plots
    plotter = Plotter(analysis)

    plotter.murakami_plot()
    plotter.anova_window_plot()
    plotter.polar_grid()
    plotter.response_map()
    plotter.sftf_fit()
    plotter.traceplot()
