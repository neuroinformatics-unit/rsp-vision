from rsp_vision.dashboard.app import get_app

if __name__ == "__main__":
    app = get_app()
    app.run_server(debug=True)