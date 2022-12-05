from pathlib import Path

from load_suite2p import read_config


def test_read_config():
    config_path = Path(__file__).parents[2] / Path(
        "load_suite2p/config/config.yml"
    )

    config = read_config.read(config_path)
    print(config)

    assert "parser" in config
    assert "server" in config
    assert "paths" in config
