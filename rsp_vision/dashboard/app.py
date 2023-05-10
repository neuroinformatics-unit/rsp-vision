import dash_bootstrap_components as dbc
from dash import Dash


def get_app() -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
        use_pages=True,
    )

    return app
