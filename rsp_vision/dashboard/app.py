import dash_bootstrap_components as dbc
from dash import Dash, html, page_container


def get_app() -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
        use_pages=True,
    )

    app.layout = dbc.Container(
        [
            html.H1(
                "RSP Vision Dashboard",
                style={"textAlign": "center", "marginBottom": "50px"},
            ),
            page_container,
        ]
    )

    return app
