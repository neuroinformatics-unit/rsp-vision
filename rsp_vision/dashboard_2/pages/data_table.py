import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback, dash_table, dcc

dash.register_page(__name__, path="/")

columns = [
    {"name": i, "id": i}
    for i in ["Element Position", "Element Name", "Symbol", "Atomic Mass"]
]
data = [
    {
        "Element Position": "6",
        "Element Name": "Carbon",
        "Symbol": "C",
        "Atomic Mass": "12.011",
    },
    {
        "Element Position": "7",
        "Element Name": "Nitrogen",
        "Symbol": "N",
        "Atomic Mass": "14.007",
    },
    {
        "Element Position": "39",
        "Element Name": "Yttrium",
        "Symbol": "Y",
        "Atomic Mass": "88.906",
    },
]

layout = dash.html.Div(
    [   
        dmc.Title(
            "Data Table",
            order=2,
            className="page-title",
        ),
        dmc.Grid(
            children=[
                dmc.Text(
                    id="selected_data_str",
                    className="selected-data-text",
                ),
                dmc.Button(
                    "Load Data",
                    id="button",
                    className="load-data-button",
                    n_clicks=0,
                ),
            ],
            className="selected-data-container",
        ),
        dmc.Container(
            children=[
                dash_table.DataTable(
                    id="table",
                    columns=columns,
                    data=data,
                    editable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    row_selectable="single",
                    row_deletable=False,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=20,
                ),
            ],
            className="table",
        ),
        dcc.Store(id="store"),
        dcc.Location(id="redirect"),
    ],
    className="page",
)


@callback(
        [Output("selected_data_str", "children"), 
         Output("store", "data"),
         Output("button", "disabled")],
        Input("table", "selected_rows")
    )
def update_graphs(selected_rows):
    if selected_rows is None or len(selected_rows) == 0:
        return "Select data to be loaded", [], True
    else:
        return f'Selected data: {data[selected_rows[0]]["Element Name"]}', data[selected_rows[0]], False

@callback(
        Output("redirect", "pathname"),
        Input("button", "n_clicks")
    )
def redirect(n_clicks):
    if n_clicks is None or n_clicks == 0:
        pass
    else:
        return "/murakami_plot"
        