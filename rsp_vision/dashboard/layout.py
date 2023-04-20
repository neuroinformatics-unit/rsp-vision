from dash import dcc, html

from rsp_vision.dashboard.styles import SIDEBAR_STYLE


def get_sidebar(responsive_rois, rois, directions):
    sidebar = html.Div(
        [
            html.H1("RSP vision"),
            html.P("Folder loaded is: AK_1111739_hL_RSPd_monitor_front"),
            html.P("Responsive ROIs"),
            html.Ul([html.Li(x + 1) for x in responsive_rois]),
            html.P("Choose a ROI"),
            dcc.Dropdown(
                options=[
                    {"label": str(roi + 1), "value": roi} for roi in rois
                ],
                value=7,
                id="demo-dropdown",
            ),
            html.P("Directions"),
            dcc.RadioItems(
                [{"label": str(dir), "value": dir} for dir in directions],
                90.0,
                id="directions-checkbox",
            ),
        ],
        style=SIDEBAR_STYLE,
    )
    return sidebar
