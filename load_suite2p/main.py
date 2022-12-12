from analysis.spatial_freq_temporal_freq import (
    SpatialFrequencyTemporalFrequency,
)
from load.load_data import load_data
from objects.photon_data import PhotonData
from plots.plotter import Plotter
from rich.prompt import Prompt

from .utils import exception_handler, start_logging


@exception_handler
def main():
    """Main function of the package. It starts logging, reads the
    configurations, asks the user to input the folder name and then
    instantiates a :class:`FolderNamingSpecs` object.
    """
    # pipeline draft
    start_logging()

    # TODO: add TUI or GUI fuctionality to get input from user
    folder_name = Prompt.ask("Please provide the folder name")

    # load data
    config, data = load_data(folder_name)

    # make analysis object
    photon_data = PhotonData(data, config)
    analysis = SpatialFrequencyTemporalFrequency(photon_data, config)

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
