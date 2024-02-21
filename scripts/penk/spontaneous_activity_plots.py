import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

local_path = Path(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/penk_analysis/"
)
local_plots_path = Path(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/penk_analysis/figures/"
)

with open(local_path / "activity.pickle", "rb") as f:
    activity = pickle.load(f)


#  heatmap for each dataset

for datagroup in activity.keys():
    for dataset in activity[datagroup].keys():
        min = 0
        max = 900
        sns.heatmap(
            activity[datagroup][dataset]["baseline"], vmin=min, vmax=max
        )
        plt.savefig(
            local_plots_path / f"{dataset}_baseline_heatmap.png", dpi=200
        )
        plt.close()
        sns.heatmap(
            activity[datagroup][dataset]["response"], vmin=min, vmax=max
        )
        plt.savefig(
            local_plots_path / f"{dataset}_response_heatmap.png", dpi=200
        )
        plt.close()


#  mean activity for each neuron across all the seesion:
fig, ax = plt.subplots(5, 2, figsize=(15, 15))


for j, datagroup in enumerate(activity.keys()):
    for k, dataset in enumerate(activity[datagroup].keys()):
        mean_baseline = []
        mean_response = []
        for i in range(len(activity[datagroup][dataset]["baseline"])):
            mean_baseline.append(
                np.nanmean(activity[datagroup][dataset]["baseline"][i])
            )
            mean_response.append(
                np.nanmean(activity[datagroup][dataset]["response"][i])
            )

        ax[k, j].scatter(
            np.arange(len(mean_baseline)), mean_baseline, label="baseline"
        )
        ax[k, j].scatter(
            np.arange(len(mean_response)), mean_response, label="response"
        )
        ax[k, j].set_title(f"{dataset}; {datagroup}")
        ax[k, j].legend()
        ax[k, j].set_xlabel("Neuron")
        ax[k, j].set_ylabel("Mean of median ΔF/F response")

        #  column title according to datagroup
        # ax[0, j].set_title(datagroup)
plt.tight_layout()

# set all y lim
for i in range(5):
    for j in range(2):
        ax[i, j].set_ylim(0, 50)

#  reduce wspace
plt.subplots_adjust(wspace=0.1)
plt.savefig(local_plots_path / "mean_activity.png", dpi=200)
plt.close()


# boxplots
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 10))

for j, datagroup in enumerate(activity.keys()):
    df = pd.DataFrame()
    baseline = []
    response = []
    for k, dataset in enumerate(activity[datagroup].keys()):
        mean_baseline = []
        mean_response = []
        for i in range(len(activity[datagroup][dataset]["baseline"])):
            mean_baseline.append(
                np.nanmean(activity[datagroup][dataset]["baseline"][i])
            )
            mean_response.append(
                np.nanmean(activity[datagroup][dataset]["response"][i])
            )
        baseline += mean_baseline
        response += mean_response

    df["baseline"] = baseline
    df["response"] = response
    sns.stripplot(data=df, color="0.25", ax=ax2[j], marker=".", alpha=0.5)
    sns.boxplot(data=df, ax=ax2[j], showfliers=False)
    #  place on top a swarmplot

    ax2[j].set_title(datagroup)
    ax2[j].set_xlabel("Static (B) vs Drift (R)")
    ax2[j].set_ylabel("Mean of median ΔF/F response")
    ax2[j].set_xticks([])

# set all y lim
for j in range(2):
    ax2[j].set_ylim(-10, 50)
plt.tight_layout()
#  no upper and right axis
sns.despine()
plt.savefig(local_plots_path / "boxplots.png", dpi=200)
plt.close()


#  just boxplot penk vs non penk response
fig4, ax4 = plt.subplots(1, 1, figsize=(5, 5))
df = pd.DataFrame()
response_2 = {}
for j, datagroup in enumerate(activity.keys()):
    mean_response = []
    for k, dataset in enumerate(activity[datagroup].keys()):
        for i in range(len(activity[datagroup][dataset]["baseline"])):
            mean_response.append(
                np.nanmean(activity[datagroup][dataset]["response"][i])
            )
    response_2[datagroup] = mean_response

len_penk = len(response_2["penk"])
len_non_penk = len(response_2["non_penk"])

df["penk"] = response_2["penk"]
df["non_penk"] = response_2["non_penk"] + [np.nan] * (len_penk - len_non_penk)
sns.stripplot(data=df, color="0.25", ax=ax4, marker=".", alpha=0.5)

sns.boxplot(data=df, ax=ax4, showfliers=False)


#  place on top a swarmplot
ax4.set_title("Penk vs Non Penk")

ax4.set_ylabel("Mean of median ΔF/F response")
ax4.set_xticks([])
ax4.set_ylim(-10, 50)
plt.tight_layout()
#  no upper and right axis
sns.despine()
plt.savefig(local_plots_path / "boxplot_penk_vs_non_penk.png", dpi=200)
plt.close()


#  scatterplot baseline vs response

fig3, ax3 = plt.subplots(1, 2, figsize=(10, 5))


for j, datagroup in enumerate(activity.keys()):
    df = pd.DataFrame()
    baseline = []
    response = []
    for k, dataset in enumerate(activity[datagroup].keys()):
        mean_baseline = []
        mean_response = []
        for i in range(len(activity[datagroup][dataset]["baseline"])):
            mean_baseline.append(
                np.nanmean(activity[datagroup][dataset]["baseline"][i])
            )
            mean_response.append(
                np.nanmean(activity[datagroup][dataset]["response"][i])
            )
        baseline += mean_baseline
        response += mean_response

    df["baseline"] = baseline
    df["response"] = response

    sns.scatterplot(
        data=df, x="baseline", y="response", ax=ax3[j], alpha=0.5, markers="."
    )
    ax3[j].set_title(datagroup)
    ax3[j].set_xlabel("Mean of median ΔF/F baseline")
    ax3[j].set_ylabel("Mean of median ΔF/F response")
    ax3[j].set_xlim(-10, 50)
    ax3[j].set_ylim(-10, 50)

plt.tight_layout()
#  no upper and right axis
sns.despine()
plt.savefig(local_plots_path / "scatterplot.png", dpi=200)
plt.close()
