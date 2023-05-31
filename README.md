# RSP vision
Tools for analyzing responses of visual neurons to drifting gratings, based on the original work of Lee Cossell.

## Installation

Clone the repository and install the dependencies in your environment with:
```bash
git clone https://github.com/neuroinformatics-unit/rsp-vision.git
cd rsp-vision
pip install .
```

## Analyse Data
### Prepare the config file and gather raw data
To test the functionalities that have been implemented so far, request the pre-processed data and store it locally in a folder called `allen_dff`.

Second, set up the environmental variables and the config file by executing:
```bash
python3 setup_for_demo.py
```
Then edit the config file with the correct paths to the data by overwriting `/path/to/`. The only path that matters at this stage is the `allen_dff` path, which should point to the folder where you stored the pre-processed data.

Finally, run `python3 demo_cli.py` to run the analysis. The script will create a file containing the analysis output which will then be used by the dashboard.

### Data processing

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

`stim` is the key of a dictionary containing information about the stimuli, with `stim["stimulus"]` being the most important. It contains the sequence of randomized features of the gratings. A given stimulus, composed of `sf`/`tf`/`direction` features, is repeated three times a day, and distributed across sessions.

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
`frames_id` corresponds to the frame number in the original data, starting from 0 for every session. `signal` is the fluorescence trace of the cell (pre-computed as dF/F), taken directly from the `f` matrix of `data_raw`. `roi_id` is the cell's ID, `session_id` is the session ID, and `sf`, `tf`, `direction` are the stimulus features. `stimulus_onset` is a boolean indicating whether the frame is the onset of the stimulus or not. Stimulus feature cells are filled only when `stimulus_onset` is True. The `PhotonData` object performs the intersection between the original `f` matrix and the `stim` dictionary in the `fill_up_with_stim_info` method. The indexes that make this join operation possible are the `stimulus_start_frames`.

Overall, the `PhotonData` object provides a more organized and accessible format for analyzing the raw data.

### Spatial and temporal frequency analysis

The goal of this analysis is to identify the cells that respond to the drifting gratings. `FrequencyResponsiveness` is the class that handles this analysis.

Since we are interested in the speed-tuning properties of neurons, we want to calculate the responses of single ROIs to combinations of `sf` and `tf` features. This is why we focus on the `sf` and `tf` columns of the `signal` dataframe. Combinations of these two features are repeated n times, where `n = len(directions) * len(repetitions)`.

In order to compute the various statistical analyses, two frame windows are taken into account, the response window, in the drifting grating part, and the baseline window, in the static or grey part of the stimulus. The mean is computed across the frames in these windows, and the difference between the two means is computed. These values are stored in the `response`, `baseline` and `subtracted` columns of the `signal` `pandas.DataFrame` in the `PhotonData` object.

Three non-parametric statistics are computed to quantify whether the response to `sf`/`tf` combinations is significant:
- The Kruskal-Wallis test
- The Wilcoxon rank-sum test
- The sign-rank test

The magnitude is also computed for each `sf`/`tf` combination. The magnitude is the difference between the mean of the median dF/F in the response window and the mean of the median dF/F in the baseline window, divided by the standard deviation of the median of the baseline window. It is computed for each repetition of the same `sf`/`tf` combinations.
For each `sf`/`tf` combination:
$$magnitude = \dfrac{mean(median(response)) - mean(median(baseline))}{std(baseline)}$$
$`median(response)`$ is the median value across all the traces that have the same `sf`/`tf` combinations. If the direction of the grating changed is irrelevant.
All the "mean of medians" are stored for each `sf`/`tf` combination in the `magnitude_over_medians` `pd.DataFrame` of `PhotonData` object.


All statistical tests and magnitude calculations are done by pooling together all directions of the same `sf`/`tf` combination (24, 48 or 72 repetitions - 1, 2 or 3 days of recording).

The threshold for the KW test and the magnitude is stored in the configuration file. The threshold for the Wilcoxon and sign-rank tests is set to 0.05. A ROI is significantly responsive if it passes the KW test and if its magnitude is above the threshold. The other two tests are currently not used.

### Response matrix and Gaussian fitting

For each ROI we can identify a response matrix that we call `median_subtracted_response`. It is a n_sf x n_tf (6x6) matrix, where `n_sf` and `n_tf` are the number of unique spatial and temporal frequencies. Each element of the matrix is the median-subtracted response of the ROI to the corresponding `sf`/`tf` combination. It is subtracted because we are interested in the difference between the response to the drifting grating and the response to the static or gray part of the stimulus. The median is computed across the repetitions of the same `sf`/`tf` combination, considering different directions independently (for a total of 3, 6 or 9 repetitions) or by pooling all directions together.

Each matrix can be fitted with a 2D elliptical Gaussian function, adjusted to incorporate ξ, the skew of the temporal frequency tuning curve, which allows us to take into account the tuning for speed.

Gaussian fitting calculations are performed by static methods in the `gaussian_calculations.py` file. It is used by the `FrequencyResponsiveness` class to pre-compute the fits for each ROI to be displayed by the dashboard. Since the fits are pre-computed, the dashboard will be able to run smoothly.

After the fit, the Gaussian is sampled to generate a 6x6 and a 100x100 matrix, which will be displayed in the dashboard, together with the median subtracted response matrix.

### Schema of the current analysis pipeline
A schematic representation of the process is shown below:
![Responsiveness analysis diagram](https://raw.githubusercontent.com/neuroinformatics-unit/rsp-vision/565b6ef3288cc5bec37341c796f11bd5e185c61a/docs/Responsiveness%20analysis%20diagram.png?token=AG642BSFJXAYISC6PI73XGDEN5QPY)
