import logging

from .read_config import read

logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="w",
    format="%(asctime)s|%(process)d|%(name)s|%(levelname)s|%(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = read()
    print(config)


if __name__ == "__main__":
    main()
