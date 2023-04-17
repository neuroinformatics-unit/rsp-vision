import numpy as np
import pandas as pd


def test_generate_report_for_statistical_tests(response):
    count_significance = pd.DataFrame(
        columns=[
            "Kruskal-Wallis test",
            "Sign test",
            "Wilcoxon signed rank test",
            "response magnitude",
            "significant_rois_positive",
            "significant_rois",
        ]
    )

    seeds = list(range(40))
    for seed in seeds:
        _response = response(seed)
        _response.calculate_mean_response_and_baseline()

        p_values = pd.DataFrame()

        p_values["Kruskal-Wallis test"] = _response.nonparam_anova_over_rois()
        count_significance.loc[seed, "Kruskal-Wallis test"] = len(
            p_values[
                p_values["Kruskal-Wallis test"]
                < _response.data.config["anova_threshold"]
            ]
        )

        (
            p_values["Sign test"],
            p_values["Wilcoxon signed rank test"],
        ) = _response.perform_sign_tests()
        count_significance.loc[seed, "Sign test"] = len(
            p_values[p_values["Sign test"] < 0.5]
        )
        count_significance.loc[seed, "Wilcoxon signed rank test"] = len(
            p_values[p_values["Wilcoxon signed rank test"] < 0.5]
        )

        magnitude = _response.response_magnitude()
        count_significance.loc[seed, "response magnitude"] = len(
            np.where(
                magnitude.groupby("roi").magnitude.max()
                > _response.data.config["response_magnitude_threshold"]
            )[0].tolist()
        )

        _response.data.config["consider_only_positive"] = False
        significant_rois_positive = _response.find_significant_rois(
            p_values, magnitude
        )
        count_significance.loc[seed, "significant_rois"] = len(
            significant_rois_positive
        )

        _response.data.config["consider_only_positive"] = True
        significant_rois = _response.find_significant_rois(p_values, magnitude)
        count_significance.loc[seed, "significant_rois_positive"] = len(
            significant_rois
        )

    proportions = count_significance.mean() / len(seeds)

    proportions = {}
    for index, row in count_significance.iterrows():
        proportions["Kruskal-Wallis test"] = count_significance[
            "Kruskal-Wallis test"
        ].sum() / (len(seeds) * _response.data.n_roi)
        proportions["Sign test"] = count_significance["Sign test"].sum() / (
            len(seeds) * _response.data.n_roi
        )
        proportions["Wilcoxon signed rank test"] = count_significance[
            "Wilcoxon signed rank test"
        ].sum() / (len(seeds) * _response.data.n_roi)
        proportions["response magnitude"] = count_significance[
            "response magnitude"
        ].sum() / (len(seeds) * _response.data.n_roi)
        proportions["significant_rois_positive"] = count_significance[
            "significant_rois_positive"
        ].sum() / (len(seeds) * _response.data.n_roi)
        proportions["significant_rois"] = count_significance[
            "significant_rois"
        ].sum() / (len(seeds) * _response.data.n_roi)

    # Generate report
    report = f"# Statistical Test Report\n\n"

    report += "## `find_significant_rois` Method\n\n"
    report += "The `find_significant_rois` method identifies regions of interest (ROIs) that show significant differences in response to stimuli based on statistical tests and a response magnitude threshold. The method takes two inputs: `p_values` and `magnitude_over_medians`.\n\n"
    report += "The method first identifies significant ROIs based on the Kruskal-Wallis test, and then identifies ROIs with a response magnitude above the `response_magnitude_threshold` specified in the `config` attribute. If the `consider_only_positive` option is set to True in the `config` attribute, the method also requires ROIs to show a significant positive response according to the Wilcoxon signed rank test.\n\n"
    report += "The method returns a set of indices of the significant ROIs in both the Kruskal-Wallis and magnitude datasets.\n\n"

    report += "## Random Data Generation\n\n"
    report += "Random data is generated using NumPy's `poisson` function. For each seed, the function generates a 3D array with dimensions `(n_sessions, n_roi, len_session)` that represents the fluorescence signal in two-photon microscopy. The values in the array are integers drawn from a Poisson distribution with an expected value that is randomly generated between 0 and 50 for each seed.\n\n"
    report += "After generating the Poisson-distributed data, the sign of each value is randomly flipped to create a more diverse set of data.\n\n"

    report += f"## Proportions of Significant Results\n\n"

    report += (
        pd.DataFrame.from_dict(
            proportions, orient="index", columns=["Proportion"]
        )
        .applymap(
            lambda x: f"<span style='color:red'>{x:.2%}</span>"
            if x > 0.05
            else f"{x:.2%}"
        )
        .to_markdown()
        + "\n"
    )

    report += f"## Count of Significant Results\n\n"
    report += (
        count_significance[
            [
                "Kruskal-Wallis test",
                "Sign test",
                "Wilcoxon signed rank test",
                "response magnitude",
                "significant_rois_positive",
                "significant_rois",
            ]
        ].to_markdown()
        + "\n\n"
    )

    # Save report as md file
    with open("docs/report.md", "w") as f:
        f.write(report)
