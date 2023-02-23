from pathlib import Path

import yaml

if __name__ == "__main__":
    env_path = Path(".env")
    with env_path.open("w", encoding="utf-8") as f:
        f.write('CONFIG_PATH="config/config.yml"')

    # create config folder
    Path("rsp_vision/config").mkdir(parents=True, exist_ok=True)

    #  create config file and store it in config folder
    f = open("rsp_vision/config/config.yml", "x")

    with open("rsp_vision/config/config.yml", "w") as f:
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
            "n_sf": 6,
            "n_tf": 6,
            "n_dir": 8,
            "trigger_interval_s": 2.5,
        }

        yaml.dump(content, f)
