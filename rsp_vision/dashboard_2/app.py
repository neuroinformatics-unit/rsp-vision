import dash
import dash_mantine_components as dmc
from dash import Dash, html

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
    external_stylesheets=external_stylesheets
)
# fmt: on

header = dmc.Header(
    height=70,
    children=[
        dmc.Title(
            "RSP Vision",
            order=1,
            className="main-title",
        )
    ],
    className="header",
)

app.layout = html.Div(
    [
        header,
        dash.page_container,
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True)
