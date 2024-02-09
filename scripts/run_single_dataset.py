import sys

from rsp_vision.console_application.app import analyse_specific_dataset

# Access the list of filenames passed as arguments
dataset = sys.argv[1:][0]

analyse_specific_dataset(dataset)
