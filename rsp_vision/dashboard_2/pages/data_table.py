import dash
import dash_mantine_components as dmc

dash.register_page(__name__, path="/")

layout = dash.html.Div(
    [
        dmc.Title("Data Table", order=2),
    ]
)
