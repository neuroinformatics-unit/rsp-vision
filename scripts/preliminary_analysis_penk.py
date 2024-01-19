#  warnings off
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


penk_non_penk = pd.read_csv(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/merged_dataset_penk.csv"
)

only_responsive = penk_non_penk[penk_non_penk["is_responsive"] == 1]


#  Plot 1, number of responsve ROIs
#  for each dataset how many responsive ROIs are there?
all_penk_responsive_count = pd.DataFrame(
    columns=["penk", "percentage", "dataset_name"],
)
penk_responsive_count = (
    only_responsive[only_responsive["penk"] == 1]
    .groupby("dataset_name")
    .count()["roi_id"]
)
print(f"Total responsive ROIs penk: {penk_responsive_count.sum()}")
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
print(f"Total responsive ROIs non penk: {non_penk_responsive_count.sum()}")
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


# barplots penk vs non penk based on dataset
g = sns.barplot(
    all_penk_responsive_count,
    x="penk",
    y="percentage",
    hue="dataset_name",
)

g.set(xlabel="Penk", ylabel="Percentage of responsive ROIs")
g.set_xticklabels(["Non Penk", "Penk"])
#  make lengend smaller
g.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

g.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_responsive_count.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# Plot 2, distplot between responsiveness and goodness of fit

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
)
#  rename subplots titles
for ax in h.axes.flat:
    if ax.get_title() == "is_responsive = 0":
        ax.set_title("Non responsive ROIs")
    else:
        ax.set_title("Responsive ROIs")

h.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_responsive_vs_fit.png",
    dpi=100,
    bbox_inches="tight",
)
plt.close()


# Fig 3: preferred sf / tf scatterplot

i = sns.jointplot(
    only_responsive,
    x="preferred_sf",
    y="preferred_tf",
    hue="penk",
    # size = 'fit_correlation',
    alpha=0.5,
    xlim=(0.01, 0.32),
    ylim=(0.5, 16),
)

# axis log
# i.set(xscale='log', yscale='log')

#  set x and y lim
# i.set(xlim=(0.01, 0.32), ylim=(0.5, 16))

i.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_preferred_sf_tf.png",
    dpi=100,
    bbox_inches="tight",
)
plt.close()

# Fig 4: preferred sf / tf 2d distplot

j = sns.displot(
    only_responsive,
    x="preferred_sf",
    y="preferred_tf",
    hue="penk",
    kind="kde",
    # bw_adjust=.25,
    # facet_kws=dict(sharex=False),
    # levels=5,
    # alpha=0.5,
)

# axis log
# j.set(xscale='log', yscale='log')
#  set x lim and y lim
# j.set(xlim=(0.01, 0.32), ylim=(0.5, 16))

plt.xlim(0.01, 0.32)
plt.ylim(0.5, 16)

j.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_preferred_sf_tf_distplot.png",
    dpi=100,
    bbox_inches="tight",
)
plt.close()

#  Fig 5: same distplot but based on dataset
k = sns.displot(
    only_responsive,
    x="preferred_sf",
    y="preferred_tf",
    hue="dataset_name",
    # kind='kde',
    facet_kws=dict(sharex=False),
    # levels=1,
    alpha=0.5,
)

# axis log
# k.set(xscale='log', yscale='log')

k.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_preferred_sf_tf_distplot_dataset.png",
    dpi=100,
    bbox_inches="tight",
)
plt.close()

# Fig 6: one dimensional distplot for tuning

only_high_correlation = only_responsive[
    only_responsive["fit_correlation"] > 0.3
]
only_tuned = only_high_correlation[
    (only_high_correlation["exponential_factor"] > 0)
    # & (only_high_correlation['exponential_factor'] < 2.5)
]

#  remove those in which sigma = 4, looks like an artifact
only_tuned = only_tuned[only_tuned["sigma_sf"] < 3.9]
only_tuned = only_tuned[only_tuned["sigma_tf"] < 3.9]

print(f"Total number of tuned ROIs: {len(only_tuned)}")
print(f"Total tuned ROIs penk: {len(only_tuned[only_tuned['penk'] == 1])}")
print(f"Total tuned ROIs non penk: {len(only_tuned[only_tuned['penk'] == 0])}")

ll = sns.displot(
    only_tuned,
    x="exponential_factor",
    hue="penk",
    kind="kde",
    bw_adjust=0.25,
    fill=True,
    aspect=1.5,
    facet_kws=dict(sharex=False),
)

ll.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_tuning_distplot.png",
    dpi=100,
    bbox_inches="tight",
)
plt.close()

# Fig 7: distplot comparing sigma tf and sigma sf

m = sns.jointplot(
    only_tuned,
    x="sigma_tf",
    y="sigma_sf",
    hue="penk",
    # size = 'exponential_factor',
    # kind='kde',
    # facet_kws=dict(sharex=False),
    # levels=10,
    alpha=0.5,
)

m.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_sigma_tf_vs_sigma_sf_distplot.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# compare the two distributions

penk_sigma_tf = only_tuned[only_tuned["penk"] == 1]["sigma_tf"]
non_penk_sigma_tf = only_tuned[only_tuned["penk"] == 0]["sigma_tf"]

penk_sigma_sf = only_tuned[only_tuned["penk"] == 1]["sigma_sf"]
non_penk_sigma_sf = only_tuned[only_tuned["penk"] == 0]["sigma_sf"]

#  compare the two distributions
# stats.ks_2samp(penk_sigma_tf, non_penk_sigma_tf)

# Fig 8: sigmas vs exponential factors


# put them together
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

# save the figure
fig.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_exp_sigma_tf_vs_sigma_sf_distplot.png",
    dpi=100,
    bbox_inches="tight",
)

#  Fig 9: peak response vs exponential factor

n = sns.jointplot(
    only_tuned,
    x="exponential_factor",
    y="peak_response",
    hue="penk",
    # size = 'exponential_factor',
    # kind='kde',
    # facet_kws=dict(sharex=False),
    # levels=10,
    alpha=0.5,
)

n.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_peak_response_vs_exp_distplot.png",
    dpi=100,
    bbox_inches="tight",
)


# Fig 10: peak response vs sigma tf

o = sns.jointplot(
    only_tuned,
    x="sigma_tf",
    y="peak_response",
    hue="penk",
    # size = 'exponential_factor',
    # kind='kde',
    # facet_kws=dict(sharex=False),
    # levels=10,
    alpha=0.5,
)

o.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_peak_response_vs_sigma_tf_distplot.png",
    dpi=100,
    bbox_inches="tight",
)

# Fig 11: peak response vs sigma sf

p = sns.jointplot(
    only_tuned,
    x="sigma_sf",
    y="peak_response",
    hue="penk",
    # size = 'exponential_factor',
    # kind='kde',
    # facet_kws=dict(sharex=False),
    # levels=10,
    alpha=0.5,
)

p.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_peak_response_vs_sigma_sf_distplot.png",
    dpi=100,
    bbox_inches="tight",
)


# Fig 12: compare sigma tf and sigma sf only of the very tuned ones

very_tuned = only_tuned[only_tuned["exponential_factor"] > 0.8]
print(f"Total number of very tuned ROIs: {len(very_tuned)}")
print(f"Total very tuned penk: {len(very_tuned[very_tuned['penk'] == 1])}")
print(f"Total very tuned non penk: {len(very_tuned[very_tuned['penk'] == 0])}")

q = sns.jointplot(
    very_tuned,
    x="sigma_tf",
    y="sigma_sf",
    hue="penk",
    # size = 'exponential_factor',
    # kind='kde',
    # facet_kws=dict(sharex=False),
    # levels=10,
    alpha=0.5,
)

q.figure.savefig(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_sigma_tf_vs_sigma_sf_distplot_very_tuned.png",
    dpi=100,
    bbox_inches="tight",
)

# Fig 13: compare sigma tf and sigma sf across datasets

fig, ax = plt.subplots(2, 5, sharey=True)

for i, dataset in enumerate(only_tuned["dataset_name"].unique()):
    color = (
        "blue"
        if only_tuned[only_tuned["dataset_name"] == dataset]["penk"].iloc[0]
        == 0
        else "orange"
    )
    if i >= 4:
        i += 1
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
    "/Users/lauraporta/local_data/rsp_vision/derivatives/figures/penk_vs_non_penk_sigma_tf_vs_sigma_sf_distplot_across_datasets.png",
    dpi=200,
    bbox_inches="tight",
)
