# This file is used to create the Dash app and the scaffold layout.
import dash
import dash_mantine_components as dmc
from dash import Dash, dcc, html

# I am using Open Sans font from Google Fonts for the whole app.
# The design of the app is controlled by the CSS file in
# rsp_vision/dashboard/assets/style.css.
google_fonts_link = (
    "https://fonts.googleapis.com/css2?family=Open+Sans&display=swap"
)
external_stylesheets = [
    {
        "href": google_fonts_link,
        "rel": "stylesheet",
    }
]
# fmt: off
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
# fmt: on

header = dmc.Header(
    height=70,
    children=[
        dmc.Title(
            "RSP vision 👁️",
            order=1,
            className="main-title",
        )
    ],
    className="header",
)

# Here I define the layout of the app. The layout is a composition of
# Mantine components and Dash components.
# Importantly, here I define the `Store` component, which is fundamental
# to share information between pages.
app.layout = html.Div(
    [
        dcc.Store(id="store", data={}),
        header,
        dash.page_container,
    ]
)
