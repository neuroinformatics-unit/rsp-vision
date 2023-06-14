import dash
import dash_mantine_components as dmc
from dash import Dash, dcc, html


def get_app() -> Dash:
    link = "https://fonts.googleapis.com/css2?family=Open+Sans&display=swap"
    external_stylesheets = [
        {
            "href": link,
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

    app.layout = html.Div(
        [
            dcc.Store(id="store", data={}),
            header,
            dash.page_container,
        ]
    )

    return app
