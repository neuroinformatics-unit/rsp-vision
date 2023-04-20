from pathlib import Path

import numpy as np
import yaml

if __name__ == "__main__":
    env_path = Path(".env")
    with env_path.open("w", encoding="utf-8") as f:
        f.write('CONFIG_PATH="config/config.yml"')

    # create config folder
    config_folder_path = Path("rsp_vision/config")
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
            "fps_tree_photon": 6.74,
            "n_sf": 6,
            "n_tf": 6,
            "n_dir": 8,
            "trigger_interval_s": 2.5,
            "anova_threshold": 0.0005,
            "response_magnitude_threshold": 2.7,
            "consider_only_positive": False,
            "only_positive_threshold": 0.05,
            "baseline": "static",
            "fitting": {
                "power_law_exp": 1,
                "lower_bounds": [-200, 0, 0, 0.01, 0.01, -np.inf],
                "upper_bounds": [np.inf, 20, 20, 4, 4, np.inf],
                "iterations_to_fit": 20,
                "jitter": 0.1,
                "oversampling_factor": 100,
            },
        }

        yaml.dump(content, f)
