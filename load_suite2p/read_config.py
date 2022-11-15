import yaml
from path import Path

config_path = Path(__file__).parent / "config/config.yml"


def read(config_path=config_path):

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
