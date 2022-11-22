import yaml
from path import Path

config_path = Path(__file__).parent / "config/config.yml"


def read(config_path: Path = config_path):
    """Reads the configuration file and returns the content as a dictionary.

    :param config_path:     path to the configuration file
    :type config_path: Path
    :return:                content of the configuration file
    :rtype: dict
    """

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
