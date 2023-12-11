from rsp_vision.console_application.app import cli_entry_point_local

if __name__ == "__main__":
    if callable(cli_entry_point_local):
        cli_entry_point_local()
