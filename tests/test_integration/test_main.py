from load_suite2p import main


def test_read_configurations():
    configs = main.read_configurations()
    assert configs is not None

    assert isinstance(configs, dict)
