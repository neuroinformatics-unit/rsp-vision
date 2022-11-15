import logging

from .read_config import read

logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="w",
    format="%(asctime)s|%(process)d|%(name)s|%(levelname)s|%(message)s",
)
logger = logging.getLogger(__name__)


def read_configurations():
    """Read configurations regarding experiment and analysis.

    :return: dictionary with configurations
    :rtype: dict
    """
    logger.info("Reading configurations")
    config = read()
    logger.info(f"Configurations read: {config}")

    return config
