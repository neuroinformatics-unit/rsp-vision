from types import ModuleType

from load_suite2p import utils


def test_get_module_for_logging():
    """Test that the module name is correctly retrieved."""
    assert isinstance(utils.get_module_for_logging(), ModuleType)
