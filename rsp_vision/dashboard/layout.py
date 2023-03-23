from dash import dcc, html

from rsp_vision.dashboard.styles import SIDEBAR_STYLE


def get_sidebar(responsive_rois, rois, directions):
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
                id="demo-dropdown",
            ),
            html.Br(),
            html.H3("Directions"),
            dcc.RadioItems(
                [{"label": str(dir), "value": dir} for dir in directions],
                90.0,
                id="directions-checkbox",
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
        ],
        style=SIDEBAR_STYLE,
    )
    return sidebar
