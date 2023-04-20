import logging
from math import log2
from multiprocessing import Pool
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss

from rsp_vision.analysis.gaussians_calculations import (
    create_gaussian_matrix,
    fit_2D_gaussian_to_data,
)
from rsp_vision.objects.photon_data import PhotonData


class FrequencyResponsiveness:
    """
    Class for analyzing responses to stimuli with different
    spatial and temporal frequencies.
    """

    def __init__(self, data: PhotonData):
        # modify the signal dataframe and responses dataframe in place
        self.data = data

    def __call__(self) -> PhotonData:
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
        logging.info(f"Edited signal dataframe:{self.data.responses.head()}")

        self.data.p_values = pd.DataFrame(
            columns=[
                "Kruskal-Wallis test",
                "Sign test",
                "Wilcoxon signed rank test",
            ]
        )

        self.data.p_values[
            "Kruskal-Wallis test"
        ] = self.nonparam_anova_over_rois()
        (
            self.data.p_values["Sign test"],
            self.data.p_values["Wilcoxon signed rank test"],
        ) = self.perform_sign_tests()
        logging.info(f"P-values for each roi:\n{self.data.p_values}")

        self.data.magnitude_over_medians = self.response_magnitude()
        logging.info(
            "Response magnitude calculated over median:\n"
            + f"{self.data.magnitude_over_medians.head()}"
        )

        self.data.responsive_rois = self.find_significant_rois(
            self.data.p_values, self.data.magnitude_over_medians
        )
        logging.info(f"Responsive ROIs: {self.data.responsive_rois}")

        # using multiprocessing, very slow step
        logging.info("Calculating Gaussian fits...")
        self.get_all_fits()
        self.calculate_downsampled_gaussian()
        self.calculate_oversampled_gaussian(
            oversampling_factor=self.data.config["fitting"][
                "oversampling_factor"
            ]
        )
        logging.info("Gaussian fits calculated")

        return self.data

    def calculate_mean_response_and_baseline(
        self,
    ) -> None:
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
        self.data.signal.loc[self.data.stimulus_idxs, "mean_response"] = [
            np.mean(
                self.data.signal.iloc[
                    self.window_mask_response[i]
                ].signal.values
            )
            for i in range(len(self.window_mask_response))
        ]

        self.data.signal.loc[self.data.stimulus_idxs, "mean_baseline"] = [
            np.mean(
                self.data.signal.iloc[
                    self.window_mask_baseline[i]
                ].signal.values
            )
            for i in range(len(self.window_mask_baseline))
        ]

        self.data.signal["subtracted"] = (
            self.data.signal["mean_response"]
            - self.data.signal["mean_baseline"]
        )

        #  new summary dataframe, more handy
        self.data.responses = self.data.signal[
            self.data.signal["stimulus_onset"]
        ][
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
        self.data.responses = self.data.responses.reset_index()

    def get_response_and_baseline_windows(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            self.data.stimulus_idxs
            + (self.data.n_frames_per_trigger * response_start)
            + (self.data.fps * 0.5)  # ignore first 0.5s
            - 1
        )
        window_end_response = (
            self.data.stimulus_idxs
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
            self.data.stimulus_idxs
            + (self.data.n_frames_per_trigger * baseline_start)
            + (self.data.fps * 1.5)  # ignore first 1.5s
            - 1
        )
        window_end_baseline = (
            self.data.stimulus_idxs
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
                self.data.responses[self.data.responses.roi_id == roi],
                id_vars=["sf", "tf"],
                value_vars=["subtracted"],
            )

            samples = np.zeros(
                (
                    len(self.data._dir) * self.data.n_triggers_per_stimulus,
                    len(self.data.sf_tf_combinations),
                )
            )

            for i, sf_tf in enumerate(self.data.sf_tf_combinations):
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
            roi_responses = self.data.responses[
                self.data.responses.roi_id == roi
            ]

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

    def response_magnitude(self) -> pd.DataFrame:
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

        magnitude_over_medians = pd.DataFrame(
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
            for i, sf_tf in enumerate(self.data.sf_tf_combinations):
                sf_tf_idx = self.data.responses[
                    (self.data.responses.sf == sf_tf[0])
                    & (self.data.responses.tf == sf_tf[1])
                    & (self.data.responses.roi_id == roi)
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
                    responses_dir_and_reps[i, :] = self.data.signal.iloc[
                        w
                    ].signal.values

                for i, w in enumerate(b_windows):
                    baseline_dir_and_reps[i, :] = self.data.signal.iloc[
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

                magnitude_over_medians = pd.concat(
                    [magnitude_over_medians, df], ignore_index=True
                )

        return magnitude_over_medians

    def find_significant_rois(
        self, p_values: Dict[str, float], magnitude_over_medians: pd.DataFrame
    ) -> Set[int]:
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
                p_values["Kruskal-Wallis test"]
                < self.data.config["anova_threshold"]
            )[0].tolist()
        )
        sig_magnitude = set(
            np.where(
                magnitude_over_medians.groupby("roi").magnitude.max()
                > self.data.config["response_magnitude_threshold"]
            )[0].tolist()
        )

        if self.data.config["consider_only_positive"]:
            sig_positive = set(
                np.where(
                    p_values["Wilcoxon signed rank test"]
                    < self.data.config["only_positive_threshold"]
                )[0].tolist()
            )
            sig_kw = sig_kw & sig_positive

        return sig_kw & sig_magnitude

    @staticmethod
    def get_preferred_sf_tf(median_subtracted_response: pd.DataFrame) -> tuple:
        sf_0, tf_0 = median_subtracted_response["subtracted"].idxmax()
        peak_response = median_subtracted_response.loc[(sf_0, tf_0)][
            "subtracted"
        ]
        return sf_0, tf_0, peak_response

    @staticmethod
    def get_median_subtracted_response(
        responses: pd.DataFrame, roi_id: int, dir: int
    ) -> pd.DataFrame:
        median_subtracted_response = (
            responses[
                (responses.roi_id == roi_id) & (responses.direction == dir)
            ]
            .groupby(["sf", "tf"])[["subtracted"]]
            .median()
        )
        return median_subtracted_response

    @staticmethod
    def get_median_subtracted_response_2d_matrix(
        median_subtracted_response: pd.DataFrame,
        sf: np.ndarray,
        tf: np.ndarray,
    ) -> np.ndarray:
        median_subtracted_response_2d_matrix = np.zeros((len(sf), len(tf)))
        for i, s in enumerate(sf):
            for j, t in enumerate(tf):
                median_subtracted_response_2d_matrix[
                    i, j
                ] = median_subtracted_response.loc[(s, t)]["subtracted"]
        return median_subtracted_response_2d_matrix

    def get_all_fits(self) -> None:
        self.data.measured_preference = {}
        self.data.fit_output = {}
        self.data.median_subtracted_response = {}
        with Pool() as p:
            # the roi order should be preserved
            roi_fit_data = p.map(
                self.get_this_roi_fits_data, range(self.data.n_roi)
            )

            for roi_id, roi_data in enumerate(roi_fit_data):
                for key in roi_data.keys():
                    self.data.measured_preference[(roi_id, key)] = roi_data[
                        key
                    ][0]
                    self.data.fit_output[(roi_id, key)] = roi_data[key][1]
                    self.data.median_subtracted_response[
                        (roi_id, key)
                    ] = roi_data[key][2]

    def get_this_roi_fits_data(self, roi_id: int) -> dict:
        """
        Returns a dictionary with the best fit parameters for each ROI and
        direction.
        """
        roi_data = {}
        for dir in self.data._dir:
            median_subtracted_response = self.get_median_subtracted_response(
                self.data.responses, roi_id, dir
            )
            sf_0, tf_0, peak_response = self.get_preferred_sf_tf(
                median_subtracted_response
            )
            msr_array = self.get_median_subtracted_response_2d_matrix(
                median_subtracted_response, self.data._sf, self.data._tf
            )

            parameters_to_fit_starting_point = [
                peak_response,
                sf_0,
                tf_0,
                np.std(self.data._sf, ddof=1),
                np.std(self.data._tf, ddof=1),
                self.data.config["fitting"]["power_law_exp"],
            ]

            best_result = fit_2D_gaussian_to_data(
                self.data._sf,
                self.data._tf,
                msr_array,
                parameters_to_fit_starting_point,
                self.data.config,
            )

            roi_data[dir] = (
                (sf_0, tf_0, peak_response),
                best_result.x,
                msr_array,
            )
        return roi_data

    def calculate_downsampled_gaussian(self) -> None:
        self.data.downsampled_gaussian = {}
        for roi_id in range(self.data.n_roi):
            for dir in self.data._dir:
                self.data.downsampled_gaussian[
                    (roi_id, dir)
                ] = create_gaussian_matrix(
                    self.data.fit_output[(roi_id, dir)],
                    self.data._sf,
                    self.data._tf,
                )

    def calculate_oversampled_gaussian(
        self, oversampling_factor: int = 100
    ) -> None:
        self.data.oversampled_gaussian = {}
        for roi_id in range(self.data.n_roi):
            for dir in self.data._dir:
                self.data.oversampled_gaussian[
                    (roi_id, dir)
                ] = create_gaussian_matrix(
                    self.data.fit_output[(roi_id, dir)],
                    np.logspace(
                        log2(self.data._sf.min()),
                        log2(self.data._sf.max()),
                        num=oversampling_factor,
                        base=2,
                    ),
                    np.logspace(
                        log2(self.data._tf.min()),
                        log2(self.data._tf.max()),
                        num=oversampling_factor,
                        base=2,
                    ),
                )
