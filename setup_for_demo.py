from pathlib import Path

import yaml

if __name__ == "__main__":
    # create .env file and add the path to config
    f = open(".env", "x")

    with open(".env", "w") as f:
        f.write('CONFIG_PATH="config/config.yml"')

    # create config folder
    Path("load_suite2p/config").mkdir(parents=True, exist_ok=True)

    #  create config file and store it in config folder
    f = open("load_suite2p/config/config.yml", "x")

    with open("load_suite2p/config/config.yml", "w") as f:
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
        }

        yaml.dump(content, f)
