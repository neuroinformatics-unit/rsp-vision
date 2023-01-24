from textual.app import App, ComposeResult
from textual.widgets import Button, Static


class MultiPhoton_RSP_Vision(App):
    def compose(self) -> ComposeResult:
        yield Static("Welcome to Multi Photon RSP Vision!")
        yield Button("Great!", id="yes", variant="primary")


if __name__ == "__main__":
    app = MultiPhoton_RSP_Vision()
    app.run()
