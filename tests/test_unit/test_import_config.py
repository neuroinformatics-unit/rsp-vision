from load_suite2p import main


def test_read_configurations_is_dict():
    configs = main.read_configurations()
    assert isinstance(configs, dict)
