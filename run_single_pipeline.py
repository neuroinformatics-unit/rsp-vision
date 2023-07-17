import sys

from rsp_vision.console_application.app import cli_entry_point_array

# Access the list of filenames passed as arguments
file = sys.argv[1:]
cli_entry_point_array(file[0])
