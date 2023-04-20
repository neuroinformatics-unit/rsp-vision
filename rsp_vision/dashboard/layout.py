from dash import dcc, html

from rsp_vision.dashboard.plotting_helpers import (
    get_circle_coordinates,
    get_direction_plot_for_controller,
)
from rsp_vision.dashboard.styles import SIDEBAR_STYLE


def get_sidebar(responsive_rois, rois, directions):
    circle_x, circle_y = get_circle_coordinates(directions)

    sidebar = html.Div(
        [
            html.H1("RSP vision"),
            html.P("Folder loaded is: AK_1111739_hL_RSPd_monitor_front"),
            html.Br(),
            html.H3("Responsive ROIs"),
            html.Ul([html.Li(x + 1) for x in responsive_rois]),
            html.Br(),
            html.H3("Choose a ROI"),
            dcc.Dropdown(
                options=[
                    {"label": str(roi + 1), "value": roi} for roi in rois
                ],
                value=7,
                id="roi-choice-dropdown",
            ),
            html.Br(),
            html.H3("Directions"),
            html.Div(
                [
                    dcc.Graph(
                        id="directions-circle",
                        style={"height": "300px", "width": "300px"},
                        figure=get_direction_plot_for_controller(
                            directions, circle_x, circle_y, 90
                        ),
                    ),
                    dcc.Store(id="direction-store", data={"value": 90}),
                    html.Div(
                        id="selected-direction", style={"display": "none"}
                    ),
                ]
            ),
            html.Br(),
            html.H3("Murakami plot controller"),
            html.H4("Choose which ROIs to show in the Murakami plot"),
            dcc.RadioItems(
                [
                    {"label": "Show all ROIs", "value": "all"},
                    {
                        "label": "Show only responsive ROIs",
                        "value": "responsive",
                    },
                    {"label": "Show only current ROI", "value": "choosen"},
                ],
                value="responsive",
                id="which-roi-to-show-in-murakami-plot",
            ),
            html.Br(),
            html.H4("Choose plot scale"),
            dcc.RadioItems(
                [
                    {"label": "Log scale", "value": "log"},
                    {"label": "Linear scale", "value": "linear"},
                ],
                value="linear",
                id="murakami-plot-scale",
            ),
            html.Br(),
            html.H3("Polar plot controller"),
            html.H4(
                "Show gaussian fit or original median subtracted response"
            ),
            dcc.RadioItems(
                [
                    {"label": "Show gaussian fit", "value": "gaussian"},
                    {
                        "label": "Show original median subtracted response",
                        "value": "original",
                    },
                ],
                value="original",
                id="polar-plot-gaussian-or-original",
            ),
            html.Br(),
            html.H4("Show mean, median or cumulative response"),
            dcc.RadioItems(
                [
                    {"label": "Show mean response", "value": "mean"},
                    {"label": "Show median response", "value": "median"},
                    {
                        "label": "Show cumulative response",
                        "value": "cumulative",
                    },
                ],
                value="mean",
                id="polar-plot-mean-or-median-or-cumulative",
            ),
        ],
        style=SIDEBAR_STYLE,
    )
    return sidebar
