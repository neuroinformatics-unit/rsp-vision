import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

remote_path = Path("/Volumes/margrie-1/laura/")
local_plots_path = Path(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/"
)
with open(remote_path / "activity.pickle", "rb") as f:
    activity = pickle.load(f)


#  heatmap for each dataset

for datagroup in activity.keys():
    for dataset in datagroup.keys():
        sns.heatmap(activity[datagroup][dataset]["baseline"])
        sns.heatmap(activity[datagroup][dataset]["response"])
        plt.savefig(local_plots_path / f"{dataset}_heatmap.png")
        plt.close()
