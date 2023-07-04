# import plotly.graph_objects as go
# import numpy as np
# import pandas as pd
# import pytest

# from rsp_vision.dashboard.pages.murakami_plot import add_data_in_figure, call_get_gaussian_matrix_to_be_plotted
# # from rsp_vision.objects.photon_data import PhotonData

# def test_add_data_in_figure(photon_data: PhotonData):
#     # Create a sample figure object
#     fig = go.Figure()

#     # Define sample data
#     all_roi = list(range(photon_data.n_roi))
#     matrix_definition = 100
#     responsive_rois = photon_data.responsive_rois
#     fitted_gaussian_matrix = call_get_gaussian_matrix_to_be_plotted(
#         photon_data.n_roi,
#         photon_data.fit_output,
#         spatial_frequencies,
#         temporal_frequencies,
#         matrix_definition,
#     )
#     spatial_frequencies = photon_data.spatial_frequencies
#     temporal_frequencies = photon_data.temporal_frequencies

#     # Call the method being tested
#     result = add_data_in_figure(
#         all_roi,
#         fig,
#         matrix_definition,
#         responsive_rois,
#         fitted_gaussian_matrix,
#         spatial_frequencies,
#         temporal_frequencies,
#     )

#     # Assert that the method returns a valid figure object
#     assert isinstance(result, go.Figure)

#     # Assert that circles and lines have been added correctly
#     for roi_id in all_roi:
#         # Check circles
#         scatter_exists = any(
#             trace.mode == "markers"
#             # and np.allclose(trace.x, [tf, median_peaks["temporal_frequency"]])
#             # and np.allclose(trace.y, [sf, median_peaks["spatial_frequency"]])
#             and trace.marker.color == "red" if roi_id in responsive_rois else "black"
#             and trace.marker.size == 10
#             for trace in result.data
#         )
#         assert scatter_exists

#         # Check lines
#         line_exists = any(
#             trace.mode == "lines"
#             # and np.allclose(trace.x, [tf, median_peaks["temporal_frequency"]])
#             # and np.allclose(trace.y, [sf, median_peaks["spatial_frequency"]])
#             and trace.line.color == "Grey"
#             and trace.line.width == 1
#             for trace in result.data
#         )
#         assert line_exists
