from rsp_vision.console_application.app import cli_entry_point_batch

if __name__ == "__main__":
    if callable(cli_entry_point_batch):
        cli_entry_point_batch()
