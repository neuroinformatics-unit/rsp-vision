from rsp_vision.console_application.app import analysis_pipeline

if __name__ == "__main__":
    if callable(analysis_pipeline):
        analysis_pipeline()
