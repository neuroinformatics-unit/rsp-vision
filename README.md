[![Wheel](https://img.shields.io/pypi/wheel/cellfinder.svg)](https://pypi.org/project/cellfinder)
[![Development Status](https://img.shields.io/pypi/status/cellfinder.svg)](https://github.com/brainglobe/cellfinder)
[![Tests](https://img.shields.io/github/workflow/status/brainglobe/cellfinder/tests)](
    https://github.com/brainglobe/cellfinder/actions)
[![codecov](https://codecov.io/gh/brainglobe/cellfinder/branch/master/graph/badge.svg?token=s3MweEFPhl)](https://codecov.io/gh/brainglobe/cellfinder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# RSP vision
TODO: Add a description of the project

## Demo Usage
To test the functionalities that have been implemented so far, request the pre-processed data and store it locally in a folder called `allen_dff`.

Second, set up the environmental variables and the config file by executing:
```bash
python3 setup_for_demo.py
```
Then edit the config file with the correct paths to the data by overwriting `/path/to/`. The only path that matters at this stage is the `allen_dff` path, which should point to the folder where you stored the pre-processed data.

Finally, run the following commands with IPython:
```python
from rsp_vision.main import main
main()
```

This script will call the `main()` method and ask you for the name of the folder containing the data you want to analyse, which corresponds to a portion of the name of the data file.

## Data processing

The original data is stored as a nested dictionary, usually referred to as the data_raw attribute. It contains the following keys: `day`, `imaging`, `f`, `is_cell`, `r_neu`, `stim`, `trig`. For our analysis, we focus mainly on `f` and `stim`.

The `f` key holds a 3D array of fluorescence traces for all cells in the recording. These cells are identified as rois. The array has dimensions (`n_sessions`, `len_session`, `n_neurons`). In each session, there are `len_session` frames, which are subdivided into multiple triggers. Triggers can be part of a stimulus or not, and their distance from each other is constant. At the beginning and end of each session, there are "baseline triggers", while the rest are stimulus triggers. A stimulus can consist of two or three parts, signalled by triggers, in which what is displayed to the animal changes. The last part always consists of drifting gratings. If there are two parts, the drifting gratings are composed of static gratings. If there are three parts, the static gratings are preceded by a grey screen.

The total length of a session is given by the following formula:

```python
len_session = int(
        (2 * n_baseline_triggers + n_stim / n_sessions * n_triggers_per_stim)
        * n_frames_per_trigger
    )
```
where len_trigger and len_baseline_trigger are the lengths of the triggers in frames.

The `stim` key is a dictionary containing information about the stimuli, with `stim["stimulus"]` being the most important. It contains the sequence of randomized features of the gratings. A given stimulus, composed of `sf`/`tf`/`direction` features, is repeated three times a day, and distributed across sessions.

`data_raw`, is reorganized into a `pandas.DataFrame` called signal, which is stored in the `PhotonData` object. The `pandas.DataFrame` has the following columns:

```
[
    "day",
    "time from beginning",
    "frames_id",
    "signal",
    "roi_id",
    "session_id",
    "sf",
    "tf",
    "direction",
    "stimulus_onset"
]
```
`frames_id` corresponds to the frame number in the original data, starting from 0 for every session. `signal` is the fluorescence trace of the cell, taken directly from the `f` matrix of `data_raw`. `roi_id` is the cell's ID, `session_id` is the session ID, and `sf`, `tf`, `direction` are the stimulus features. `stimulus_onset` is a boolean indicating whether the frame is the onset of the stimulus or not. Stimulus feature cells are filled only when `stimulus_onset` is True. The `PhotonData` object performs the intersection between the original `f` matrix and the `stim` dictionary in the `fill_up_with_stim_info` method. The indexes that make this join operation possible are the `stimulus_start_frames`.

Overall, the `PhotonData` object provides a more organized and accessible format for analyzing the raw data.

## Spatial and temporal frequency analysis
