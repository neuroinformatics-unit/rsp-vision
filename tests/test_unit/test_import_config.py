from load_suite2p import main


def test_read_configurations_is_dict():
    configs = main.read_configurations()
    assert isinstance(configs, dict)


def test_read_configurations_has_right_keys():
    configs = main.read_configurations()
    assert configs.keys() == {
        "analysis-type",
        "region-id",
        "fluorescence_type",
    }
