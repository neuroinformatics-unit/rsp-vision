import dash
import dash_mantine_components as dmc
from dash import html

dash.register_page(__name__, path="/polar_plots")

layout = html.Div(
    [
        dmc.Title(
            "Polar plots",
            className="page-title",
        ),
        dmc.Grid(
            children=[
                dmc.Col(
                    [
                        dmc.Text(
                            id="selected_data_str_polar",
                        ),
                        html.Br(),
                        # dmc.Switch(
                        #     id="show-only-responsive",
                        #     label="Show only responsive ROIs",
                        #     checked=True,
                        # ),
                        html.Br(),
                        html.Br(),
                        dmc.NavLink(
                            label="Back to Data Table",
                            href="/",
                            className="navlink",
                        ),
                        dmc.NavLink(
                            label="Murakami plot",
                            href="/murakami_plot",
                            className="navlink",
                        ),
                        dmc.NavLink(
                            label="SF-TF facet plot and gaussians",
                            href="/sf_tf_facet_plot",
                            className="navlink",
                        ),
                        dmc.NavLink(
                            label="Polar plots",
                            href="/polar_plots",
                            className="navlink",
                            disabled=True,
                        ),
                    ],
                    span=2,
                ),
                dmc.Col(
                    [
                        html.Div(
                            id="polar-plots-grid",
                            className="sf-tf-plot",
                        ),
                    ],
                    span="auto",
                ),
                dmc.Col(
                    [
                        html.Div(
                            id="polar-plots",
                            className="sf-tf-plot",
                        ),
                    ],
                    span=3,
                ),
            ],
            className="sf-tf-container",
        ),
    ],
    className="page",
)
