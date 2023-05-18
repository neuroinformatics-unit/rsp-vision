import dash
import dash_mantine_components as dmc

dash.register_page(__name__, path="/murakami_plot")

layout = dash.html.Div(
    [
        dmc.Title("Murakami", order=2),
    ]
)
