from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

from rsp_vision.dashboard.pages.data_table import update_storage


def test_update_storage_no_row_selected():
    update_storage(None)

    assert context_value.get("selected_data_str") == "Select data to be loaded"

    