from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rsp-vision")
except PackageNotFoundError:
    # package is not installed
    pass