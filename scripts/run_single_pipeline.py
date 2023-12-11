import sys

from rsp_vision.console_application.app import cli_entry_point_array

# Access the list of filenames passed as arguments
job_id = int(sys.argv[1:][0])
cli_entry_point_array(job_id)
