from pathlib import Path

import yaml

if __name__ == "__main__":
    env_path = Path(".env")
    with env_path.open("w", encoding="utf-8") as f:
        f.write('CONFIG_PATH="config/config.yml"')

    # create config folder
    config_folder_path = Path("load_suite2p/config")
    config_folder_path.mkdir(parents=True, exist_ok=True)
    config_path = config_folder_path / "config.yml"

    with config_path.open("w", encoding="utf-8") as f:
        content = {
            "parser": "Parser2pRSP",
            "paths": {
                "imaging": "/path/to/",
                "allen-dff": "/path/to/allen_dff",
                "serial2p": "/path/to/serial2p",
                "stimulus-ai-schedule": "/path/to/stimulus_AI_schedule_files",
            },
            "use-allen-dff": "true",
            "analysis-type": "sf_tf",
            "padding": [25, 50],
            "drift_order": 2,
            "fps_two_photon": 30,
            "fps_tree_photon": 15,
        }

        yaml.dump(content, f)
