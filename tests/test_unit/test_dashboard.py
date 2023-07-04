from dash.testing.composite import DashComposite

from rsp_vision.dashboard.app import app


def test_app_starts(
    dash_duo: DashComposite,
) -> None:
    dash_duo.start_server(app)

    dash_duo.find_element("store")

    assert dash_duo.get_logs() == []

    return None
