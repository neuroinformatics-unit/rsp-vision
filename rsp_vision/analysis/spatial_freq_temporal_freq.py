import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss

from rsp_vision.analysis.utils import get_fps
from rsp_vision.objects.enums import PhotonType
from rsp_vision.objects.photon_data import PhotonData


class FrequencyAnalysis:
    """
    Class for analyzing responses to stimuli with different
    spatial and temporal frequencies.
    """

    def __init__(self, data: PhotonData, photon_type: PhotonType):
        self.data = data
        self.photon_type = photon_type
        self.fps = get_fps(photon_type, data.config)

        self.padding_start = int(data.config["padding"][0])
        self.padding_end = int(data.config["padding"][1])

        self.stimulus_idxs = self.data.signal[
            self.data.signal["stimulus_onset"]
        ].index

        self._sf = np.sort(self.data.uniques["sf"])
        self._tf = np.sort(self.data.uniques["tf"])
        self.sf_tf_combinations = np.array(
            np.meshgrid(self._sf, self._tf)
        ).T.reshape(-1, 2)
        self._dir = self.data.uniques["direction"]

        self.signal = self.data.signal
        self.signal["mean_response"] = np.nan
        self.signal["mean_baseline"] = np.nan

    def responsiveness(self):
        """
        Calculate the responsiveness of each ROI in the signal dataframe.

        This method calculates the responsiveness of each ROI in the signal
        dataframe, based on the mean response and mean baseline signals
        calculated in the calculate_mean_response_and_baseline method.
        First, it performs three different statistical tests to determine
        if the response is significantly different from the baseline:
        a Kruskal-Wallis test, a Sign test, and a Wilcoxon signed rank test.
        The resulting p-values for each ROI are stored in a pandas DataFrame
        and logged for debugging purposes.

        Next, the method calculates the response magnitude for each ROI, by
        taking the  difference between the mean response and mean baseline
        signals, and dividing by standard deviation of the baseline signal.
        The response and baseline mean are calculated over the median traces.

        Finally, the method identifies the ROIs that show significant
        responsiveness based on the results of the statistical tests.
        """
        self.calculate_mean_response_and_baseline()
        logging.info(f"Edited signal dataframe:{self.responses.head()}")

        self.p_values = pd.DataFrame(
            columns=[
                "Kruskal-Wallis test",
                "Sign test",
                "Wilcoxon signed rank test",
            ]
        )

        self.p_values["Kruskal-Wallis test"] = self.nonparam_anova_over_rois()
        (
            self.p_values["Sign test"],
            self.p_values["Wilcoxon signed rank test"],
        ) = self.perform_sign_tests()
        logging.info(f"P-values for each roi:\n{self.p_values}")

        self.magintude_over_medians = self.response_magnitude()
        logging.info(
            "Response magnitude calculated over median:\n"
            + f"{self.magintude_over_medians.head()}"
        )

        self.responsive_rois = self.find_significant_rois()
        logging.info(f"Responsive ROIs: {self.responsive_rois}")

    def calculate_mean_response_and_baseline(
        self,
    ):
        """
        Calculate the mean response and mean baseline signals for each ROI
        in the signal dataframe.

        This method calculates the mean response and mean baseline signals
        for each ROI in the signal dataframe,
        based on the response and baseline windows defined by the method
        `get_response_and_baseline_windows`, which this method calls.
        It then subtracts the mean baseline from
        the mean response to obtain the mean subtracted signal. The resulting
        mean response, mean baseline, and mean subtracted signal values are
        stored in the signal dataframe, and a subset of the dataframe
        containing only the relevant columns is stored in the responses
        attribute.

        Returns:
            None
        """
        logging.info("Start to edit the signal dataframe...")

        (
            self.window_mask_response,
            self.window_mask_baseline,
        ) = self.get_response_and_baseline_windows()

        # add calculated values in the row corresponding to
        # the startframe of every stimulus
        self.signal.loc[self.stimulus_idxs, "mean_response"] = [
            np.mean(
                self.signal.iloc[self.window_mask_response[i]].signal.values
            )
            for i in range(len(self.window_mask_response))
        ]

        self.signal.loc[self.stimulus_idxs, "mean_baseline"] = [
            np.mean(
                self.signal.iloc[self.window_mask_baseline[i]].signal.values
            )
            for i in range(len(self.window_mask_baseline))
        ]

        self.signal["subtracted"] = (
            self.signal["mean_response"] - self.signal["mean_baseline"]
        )

        #  new summary dataframe, more handy
        self.responses = self.signal[self.signal["stimulus_onset"]][
            [
                "frames_id",
                "roi_id",
                "session_id",
                "sf",
                "tf",
                "direction",
                "mean_response",
                "mean_baseline",
                "subtracted",
            ]
        ]
        self.responses = self.responses.reset_index()

    def get_response_and_baseline_windows(self):
        """
        Get the window of indices corresponding to the response and
        baseline periods for each stimulus presentation.

        Returns:
        -------
        window_mask_response : numpy.ndarray
            A 2D numpy array containing the signal dataframe indices
            for the response period for each stimulus presentation.
            Each row corresponds to a stimulus presentation and each
            column contains the frame indices in that presentation.
        window_mask_baseline : numpy.ndarray
            A 2D numpy array containing the signal dataframe indices
            for the baseline period for each stimulus presentation.
            Each row corresponds to a stimulus presentation and each
            column contains the frame indices in that presentation.
        """

        baseline_start = 0
        if self.data.n_triggers_per_stimulus == 3:
            response_start = 2
            if self.data.config["baseline"] == "static":
                baseline_start = 1
        if self.data.n_triggers_per_stimulus == 2:
            response_start = 1

        # Identify the rows in the window of each stimulus onset
        window_start_response = (
            self.stimulus_idxs
            + (self.data.n_frames_per_trigger * response_start)
            + (self.fps * 0.5)  # ignore first 0.5s
            - 1
        )
        window_end_response = (
            self.stimulus_idxs
            + (self.data.n_frames_per_trigger * (response_start + 1))
            - 1
        )
        window_mask_response = np.vstack(
            [
                np.arange(start, end)
                for start, end in zip(
                    window_start_response, window_end_response
                )
            ]
        )

        window_start_baseline = (
            self.stimulus_idxs
            + (self.data.n_frames_per_trigger * baseline_start)
            + (self.fps * 1.5)  # ignore first 1.5s
            - 1
        )
        window_end_baseline = (
            self.stimulus_idxs
            + (self.data.n_frames_per_trigger * (baseline_start + 1))
            - 1
        )
        window_mask_baseline = np.vstack(
            [
                np.arange(start, end)
                for start, end in zip(
                    window_start_baseline, window_end_baseline
                )
            ]
        )

        return window_mask_response, window_mask_baseline

    def nonparam_anova_over_rois(self) -> dict:
        """
        Perform a nonparametric ANOVA test over each ROI in the dataset.
        This test is based on the Kruskal-Wallis H Test, which compares
        whether more than two independent samples have different
        distributions. For each ROI, this method creates a table with one row
        for each combination of spatial and temporal frequencies, and one
        column for each presentation of the stimulus. Then, it applies the
        Kruskal-Wallis H Test to determine whether the distribution of
        responses across different stimuli is significantly different. The
        resulting p-values are returned as a dictionary where the keys are
        the ROI IDs and the values are the p-values for each ROI.

        Returns
        -------
        dict
            A dictionary where the keys are the ROI IDs and the values are
            the p-values for each ROI.

        """

        p_values = {}
        for roi in range(self.data.n_roi):
            roi_responses = pd.melt(
                self.responses[self.responses.roi_id == roi],
                id_vars=["sf", "tf"],
                value_vars=["subtracted"],
            )

            samples = np.zeros(
                (
                    len(self._dir) * self.data.n_triggers_per_stimulus,
                    len(self.sf_tf_combinations),
                )
            )

            for i, sf_tf in enumerate(self.sf_tf_combinations):
                # samples are each presentation of an sf/tf combination,
                # regardless of direction and repetition
                samples[:, i] = roi_responses[
                    (roi_responses.sf == sf_tf[0])
                    & (roi_responses.tf == sf_tf[1])
                ].value

            _, p_val = ss.kruskal(*samples.T)
            p_values[roi] = p_val

        return p_values

    def perform_sign_tests(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Perform sign test and Wilcoxon signed rank test on the subtracted
        response data for each ROI.

        Returns:
            A tuple containing two dictionaries with ROI IDs as keys and
            p-values as values. The first dictionary contains the p-values
            for the sign test (implemented with a binomial test) for each ROI.
            The test checks whether the proportion of positive differences
            between the response and baseline periods is greater than 0.5.
            The second dictionary contains the p-values for the Wilcoxon
            signed rank test for each ROI.
            The test checks whether the distribution of differences between
            the response and baseline periods is shifted to the right.
        """

        p_st = {}
        p_wsrt = {}
        for roi in range(self.data.n_roi):
            roi_responses = self.responses[self.responses.roi_id == roi]

            # Sign test (implemented with binomial test)
            p_st[roi] = ss.binomtest(
                sum([1 for d in roi_responses.subtracted if d > 0]),
                n=len(roi_responses.subtracted),
                alternative="greater",
            ).pvalue

            # Wilcoxon signed rank test
            _, p_wsrt[roi] = ss.wilcoxon(
                x=roi_responses.subtracted,
                alternative="greater",
            )

        return p_st, p_wsrt

    def response_magnitude(self):
        """
        Compute the response magnitude for each combination of spatial and
        temporal frequency and ROI.

        Returns:
            A Pandas DataFrame with the following columns:
                - "roi": the ROI index
                - "sf": the spatial frequency
                - "tf": the temporal frequency
                - "response_mean": the mean of the median response signal
                - "baseline_mean": the mean of the median baseline signal
                - "baseline_std": the standard deviation of the median
                    baseline signal
                - "magnitude": the response magnitude, defined as
                    (response_mean - baseline_mean) / baseline_std
        """

        magintude_over_medians = pd.DataFrame(
            columns=[
                "roi",
                "sf",
                "tf",
                "response_mean",
                "baseline_mean",
                "baseline_std",
                "magnitude",
            ]
        )

        for roi in range(self.data.n_roi):
            for i, sf_tf in enumerate(self.sf_tf_combinations):
                sf_tf_idx = self.responses[
                    (self.responses.sf == sf_tf[0])
                    & (self.responses.tf == sf_tf[1])
                    & (self.responses.roi_id == roi)
                ].index
                r_windows = self.window_mask_response[sf_tf_idx]
                b_windows = self.window_mask_baseline[sf_tf_idx]

                responses_dir_and_reps = np.zeros(
                    (r_windows.shape[0], r_windows.shape[1])
                )
                baseline_dir_and_reps = np.zeros(
                    (b_windows.shape[0], b_windows.shape[1])
                )

                for i, w in enumerate(r_windows):
                    responses_dir_and_reps[i, :] = self.signal.iloc[
                        w
                    ].signal.values

                for i, w in enumerate(b_windows):
                    baseline_dir_and_reps[i, :] = self.signal.iloc[
                        w
                    ].signal.values

                median_response = np.median(responses_dir_and_reps, axis=0)
                median_baseline = np.median(baseline_dir_and_reps, axis=0)

                m_r = np.mean(median_response)
                m_b = np.mean(median_baseline)
                std_b = np.std(median_baseline, ddof=1)
                magnitude = (m_r - m_b) / std_b

                df = pd.DataFrame(
                    {
                        "roi": roi,
                        "sf": sf_tf[0],
                        "tf": sf_tf[1],
                        "response_mean": m_r,
                        "baseline_mean": m_b,
                        "baseline_std": std_b,
                        "magnitude": magnitude,
                    },
                    index=[0],
                )

                magintude_over_medians = pd.concat(
                    [magintude_over_medians, df], ignore_index=True
                )

        return magintude_over_medians

    def find_significant_rois(self):
        """
        Returns a set of ROIs that are significantly responsive, based on
        statistical tests and a response magnitude threshold.

        The method first identifies ROIs that show significant differences
        across at least one condition using the Kruskal-Wallis test.
        ROIs with p-values below the `anova_threshold` specified in the
        `config` attribute are considered significant.

        It then identifies ROIs with a response magnitude above the
        `response_magnitude_threshold` specified in the `config` attribute.
        The response magnitude is defined as the difference between the
        mean response and mean baseline, divided by the standard deviation
        of the baseline.

        If the `consider_only_positive` option is set to True in the `config`
        attribute, the method also requires ROIs to show a significant
        positive response according to the Wilcoxon signed rank test. ROIs
        with p-values below the `only_positive_threshold` specified in the
        `config` attribute are considered significant.

        Returns:
            set: A set of ROIs that are significantly responsive based on the
            specified criteria.
        """
        sig_kw = set(
            np.where(
                self.p_values["Kruskal-Wallis test"].values
                < self.data.config["anova_threshold"]
            )[0].tolist()
        )
        sig_magnitude = set(
            np.where(
                self.magintude_over_medians.groupby("roi").magnitude.max()
                > self.data.config["response_magnitude_threshold"]
            )[0].tolist()
        )

        if self.data.config["consider_only_positive"]:
            sig_positive = set(
                np.where(
                    self.p_values["Wilcoxon signed rank test"].values
                    < self.data.config["only_positive_threshold"]
                )[0].tolist()
            )
            sig_kw = sig_kw & sig_positive

        return sig_kw & sig_magnitude

    def get_fit_parameters(self):
        # calls _fit_two_dimensional_elliptical_gaussian
        raise NotImplementedError("This method is not implemented yet")

    def get_preferred_direction_all_rois(self):
        raise NotImplementedError("This method is not implemented yet")

    def _fit_two_dimensional_elliptical_gaussian(self):
        # as described by Priebe et al. 2006
        # add the variations added by Andermann et al. 2011 / 2013
        # calls _2d_gaussian
        # calls _get_response_map
        raise NotImplementedError("This method is not implemented yet")


class Gaussian2D:
    # a 2D gaussian function
    # also used by plotting functions
    def __init__(self):
        # different kinds of 2D gaussians:
        # - 2D gaussian
        # - 2D gaussian Andermann
        # - 2D gaussian Priebe
        raise NotImplementedError("This method is not implemented yet")


class ResponseMap:
    # also used by plotting functions
    def _get_response_map(self):
        # calls _get_preferred_direction
        raise NotImplementedError("This method is not implemented yet")

    def _get_preferred_direction(self):
        raise NotImplementedError("This method is not implemented yet")
