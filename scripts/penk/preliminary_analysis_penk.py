#  warnings off
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

path_derivatives = Path("/Users/lauraporta/local_data/rsp_vision/derivatives/")
path_figures = path_derivatives / "figures"
penk_non_penk = pd.read_csv(path_derivatives / "merged_dataset_penk.csv")

min_corr = 0.65
min_zeta = 0
not_normalized = False


# =============================================================================
#  Plot 1, number of responsve ROIs

only_responsive = penk_non_penk[penk_non_penk["is_responsive"] == 1]

all_penk_responsive_count = pd.DataFrame(
    columns=["penk", "percentage", "dataset_name"],
)
penk_responsive_count = (
    only_responsive[only_responsive["penk"] == 1]
    .groupby("dataset_name")
    .count()["roi_id"]
)
for dataset in penk_responsive_count.index:
    total = only_responsive[only_responsive["dataset_name"] == dataset].iloc[
        0
    ]["total_rois_in_dataset"]
    penk_responsive_count[dataset] = penk_responsive_count[dataset] / total

    all_penk_responsive_count.loc[len(all_penk_responsive_count)] = [
        1,
        penk_responsive_count[dataset],
        dataset,
    ]

non_penk_responsive_count = (
    only_responsive[only_responsive["penk"] == 0]
    .groupby("dataset_name")
    .count()["roi_id"]
)
for dataset in non_penk_responsive_count.index:
    total = only_responsive[only_responsive["dataset_name"] == dataset].iloc[
        0
    ]["total_rois_in_dataset"]
    non_penk_responsive_count[dataset] = (
        non_penk_responsive_count[dataset] / total
    )

    all_penk_responsive_count.loc[len(all_penk_responsive_count)] = [
        0,
        non_penk_responsive_count[dataset],
        dataset,
    ]

g = sns.barplot(
    all_penk_responsive_count,
    x="penk",
    y="percentage",
    hue="dataset_name",
)

g.set(xlabel="Penk", ylabel="Percentage of responsive ROIs")
g.set_xticklabels(["Non Penk", "Penk"])
g.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

g.figure.savefig(
    path_figures / "Fig1_responsive_count.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# =============================================================================
# Plot 1.5, boxplot of the number of responsive ROIs

g = sns.boxplot(
    all_penk_responsive_count,
    x="penk",
    y="percentage",
    # hue="dataset_name",
)

g.set(xlabel="Penk", ylabel="Percentage of responsive ROIs")
g.set_xticklabels(["Non Penk", "Penk"])
g.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

g.figure.savefig(
    path_figures / "Fig1.5_responsive_count.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# =============================================================================
# Plot 2, responsive vs fit correlation

h = sns.displot(
    penk_non_penk,
    x="fit_correlation",
    hue="penk",
    # kind='kde',
    # bw_adjust=.25,
    col="is_responsive",
    fill=True,
    aspect=1.5,
    facet_kws=dict(sharex=False),
    common_norm=not_normalized,
)
for ax in h.axes.flat:
    if ax.get_title() == "is_responsive = 0":
        ax.set_title("Non responsive ROIs")
    else:
        ax.set_title("Responsive ROIs")

h.figure.savefig(
    path_figures / "Fig2_responsive_vs_fit.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# =============================================================================
# Fig 3: preferred sf / tf scatterplot, only responsive

i = sns.jointplot(
    only_responsive,
    x="preferred_sf",
    y="preferred_tf",
    hue="penk",
    alpha=0.5,
    xlim=(0.01, 0.32),
    ylim=(0.5, 16),
    marginal_kws={"common_norm": not_normalized},
)

i.figure.savefig(
    path_figures / "Fig3_preferred_sf_tf_only_responsive.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# =============================================================================
# Fig 4: preferred sf / tf 2d distplot, only responsive

j = sns.displot(
    only_responsive,
    x="preferred_sf",
    y="preferred_tf",
    hue="penk",
    kind="kde",
    common_norm=not_normalized,
)

plt.xlim(0.01, 0.32)
plt.ylim(0.5, 16)

j.figure.savefig(
    path_figures / "Fig4_preferred_sf_tf_distplot.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# =============================================================================
#  Fig 5: same distplot but based on dataset

k = sns.displot(
    only_responsive,
    x="preferred_sf",
    y="preferred_tf",
    hue="dataset_name",
    facet_kws=dict(sharex=False),
    alpha=0.5,
    common_norm=False,
)

k.figure.savefig(
    path_figures / "Fig5_preferred_sf_tf_only_responsive.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# =============================================================================
# Fig 6: one dimensional distplot for tuning

only_high_correlation = only_responsive[
    only_responsive["fit_correlation"] > min_corr
]
only_tuned = only_high_correlation[
    (only_high_correlation["exponential_factor"] > min_zeta)
    # & (only_high_correlation['exponential_factor']  < 2.5)
]

#  remove those in which sigma = 4, looks like an artifact
# only_tuned = only_tuned[only_tuned["sigma_sf"] < 3.9]
# only_tuned = only_tuned[only_tuned["sigma_tf"] < 3.9]

ll = sns.displot(
    only_tuned,
    x="exponential_factor",
    hue="penk",
    kind="kde",
    bw_adjust=0.25,
    fill=True,
    aspect=1.5,
    facet_kws=dict(sharex=False),
    common_norm=not_normalized,
)

ll.figure.savefig(
    path_figures / "Fig6_tuning_distplot_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# =============================================================================
# Fig 7: distplot comparing sigma tf and sigma sf

m = sns.jointplot(
    only_tuned,
    x="sigma_tf",
    y="sigma_sf",
    hue="penk",
    alpha=0.5,
    marginal_kws={"common_norm": not_normalized},
)

m.figure.savefig(
    path_figures / "Fig7_sigmas_comparison_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# =============================================================================
# Fig 8: sigmas vs exponential factors
# no need to normalize, scatterplot

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1 = sns.scatterplot(
    only_tuned,
    x="exponential_factor",
    y="sigma_sf",
    hue="penk",
    alpha=0.5,
    ax=ax1,
)
ax2 = sns.scatterplot(
    only_tuned,
    x="exponential_factor",
    y="sigma_tf",
    hue="penk",
    alpha=0.5,
    ax=ax2,
)

fig.savefig(
    path_figures / "Fig8_sigmas_vs_exp_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
#  Fig 9: peak response vs exponential factor

n = sns.jointplot(
    only_tuned,
    x="exponential_factor",
    y="peak_response_not_fitted",
    hue="penk",
    alpha=0.5,
    marginal_kws={"common_norm": not_normalized},
)

n.figure.savefig(
    path_figures
    / "Fig9_peak_response_not_fitted_vs_exp_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 10: peak response vs sigma tf

o = sns.jointplot(
    only_tuned,
    x="sigma_tf",
    y="peak_response_not_fitted",
    hue="penk",
    alpha=0.5,
    marginal_kws={"common_norm": not_normalized},
)

o.figure.savefig(
    path_figures
    / "Fig10_peak_response_not_fitted_vs_sigma_tf_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 11: peak response vs sigma sf

penk_nice_fit_perc = sns.jointplot(
    only_tuned,
    x="sigma_sf",
    y="peak_response_not_fitted",
    hue="penk",
    alpha=0.5,
    marginal_kws={"common_norm": not_normalized},
)

penk_nice_fit_perc.figure.savefig(
    path_figures
    / "Fig11_peak_response_not_fitted_vs_sigma_sf_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 12: compare sigma tf and sigma sf only of the very tuned ones

strict_tuning = only_tuned[
    (only_tuned["exponential_factor"] > 0.2)
    & (only_tuned["exponential_factor"] < 4)
]

q = sns.jointplot(
    strict_tuning,
    x="sigma_tf",
    y="sigma_sf",
    hue="penk",
    alpha=0.5,
    marginal_kws={"common_norm": not_normalized},
)

q.figure.savefig(
    path_figures / "Fig12_sigma_tf_vs_sigma_sf_distplot_strict_tuning.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 13: compare sigma tf and sigma sf across datasets

fig, ax = plt.subplots(2, 5, sharey=True)

for i, dataset in enumerate(only_tuned["dataset_name"].unique()):
    color = (
        "blue"
        if only_tuned[only_tuned["dataset_name"] == dataset]["penk"].iloc[0]
        == 0
        else "orange"
    )

    ax[i // 5, i % 5] = sns.scatterplot(
        only_tuned[only_tuned["dataset_name"] == dataset],
        x="sigma_tf",
        y="sigma_sf",
        color=color,
        alpha=0.5,
        ax=ax[i // 5, i % 5],
        legend=False,
    )
    plt.xlim(0, 5)
    plt.ylim(0, 5)


fig.savefig(
    path_figures / "Fig13_sigmas_across_datasets_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 14: compare sigma tf and sigma sf kde but normalized by number of ROIs
# remove?

r = sns.displot(
    only_tuned,
    x="sigma_tf",
    y="sigma_sf",
    hue="penk",
    kind="kde",
    bw_adjust=0.25,
    fill=True,
    aspect=1.5,
    facet_kws=dict(sharex=False),
    common_norm=False,
    alpha=0.5,
)

r.figure.savefig(
    path_figures / "Fig14_sigma_tf_vs_sigma_sf_distplot_normalized.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 15: sigma sf / tf ratio vs exponential factor

only_tuned["sigma_ratio"] = only_tuned["sigma_tf"] / only_tuned["sigma_sf"]
s = sns.jointplot(
    only_tuned,
    x="exponential_factor",
    y="sigma_ratio",
    hue="penk",
    alpha=0.5,
    marginal_kws={"common_norm": not_normalized},
)


s.figure.savefig(
    path_figures / "Fig15_sigma_sf_tf_ratio_vs_exp_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 16: sigma sf vs sigma tf include non responsive

all_tuned = penk_non_penk[
    (penk_non_penk["exponential_factor"] > min_zeta)
    & (penk_non_penk["fit_correlation"] > min_corr)
]

# # remove artifacts at 4
# all_tuned = all_tuned[all_tuned["sigma_sf"] < 3.9]
# all_tuned = all_tuned[all_tuned["sigma_tf"] < 3.9]

#  exclude extreme exponential factor
# all_tuned = all_tuned[all_tuned["exponential_factor"] < 10]

t = sns.jointplot(
    all_tuned,
    x="sigma_tf",
    y="sigma_sf",
    hue="penk",
    alpha=0.5,
    marginal_kws={"common_norm": not_normalized},
)

t.figure.savefig(
    path_figures
    / "Fig16_sigmas_including_not_responsive_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 17: sigma sf vs sigma tf include non responsive all datasets

fig, ax = plt.subplots(2, 5, sharey=True)

for i, dataset in enumerate(all_tuned["dataset_name"].unique()):
    color = (
        "blue"
        if all_tuned[all_tuned["dataset_name"] == dataset]["penk"].iloc[0] == 0
        else "orange"
    )

    ax[i // 5, i % 5] = sns.scatterplot(
        all_tuned[all_tuned["dataset_name"] == dataset],
        x="sigma_tf",
        y="sigma_sf",
        color=color,
        alpha=0.5,
        ax=ax[i // 5, i % 5],
        legend=False,
    )
    plt.xlim(0, 5)
    plt.ylim(0, 5)


fig.savefig(
    path_figures / "Fig17_sigmas_across_datasets_including_not_resp_HCT.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 18: distplot of peak response not fitted (like in fig 2)

u = sns.displot(
    all_tuned,
    x="peak_response_not_fitted",
    hue="penk",
    col="is_responsive",
    fill=True,
    aspect=1.5,
    facet_kws=dict(sharex=False),
    common_norm=not_normalized,
)

for ax in u.axes.flat:
    if ax.get_title() == "is_responsive = 0":
        ax.set_title("Non responsive ROIs")
    else:
        ax.set_title("Responsive ROIs")


u.figure.savefig(
    path_figures / "Fig18_peak_response_not_fitted_distplot.png",
    dpi=200,
    bbox_inches="tight",
)

# =============================================================================
# Fig 21: all tuned but sigma tf vs sigma sf colored by exponential factor


fig, ax = plt.subplots(1, 2, sharey=True)

for i, penk in enumerate([0, 1]):
    ax[i] = sns.scatterplot(
        all_tuned[all_tuned["penk"] == penk],
        x="sigma_tf",
        y="sigma_sf",
        hue="exponential_factor",
        alpha=0.5,
        ax=ax[i],
        hue_norm=(0, 1),
        legend=False,
        palette="viridis",
    )
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    #  title
    if penk == 0:
        ax[i].set_title("Non Penk")
    else:
        ax[i].set_title("Penk")

fig.savefig(
    path_figures
    / "Fig21_sigmas_vs_exp_including_not_responsive_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)


# =============================================================================
# Fig 22: same as 21 but separating in rows these groups of exponential factors


ranges = [0, 0.5, 2, 10]
all_tuned["exp_group"] = pd.cut(
    all_tuned["exponential_factor"], bins=ranges, labels=False
)

yy = sns.displot(
    all_tuned,
    x="sigma_tf",
    y="sigma_sf",
    hue="penk",
    # kind="kde",
    # bw_adjust=0.25,
    # fill=True,
    # aspect=1.5,
    col="exp_group",
    facet_kws=dict(sharex=False),
    common_norm=not_normalized,
    alpha=0.5,
)

# add titles
for ax in yy.axes.flat:
    if ax.get_title() == "exp_group = 0.0":
        ax.set_title("0 < exp < 0.5")
    elif ax.get_title() == "exp_group = 1.0":
        ax.set_title("0.5 < exp < 2")
    elif ax.get_title() == "exp_group = 2.0":
        ax.set_title("2 < exp < 10")

yy.savefig(
    path_figures
    / "Fig22_sigmas_vs_exp_including_not_responsive_high_corr_and_tuned.png",
    dpi=200,
    bbox_inches="tight",
)
