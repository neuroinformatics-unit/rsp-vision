# Statistical Test Report

## `find_significant_rois` Method

The `find_significant_rois` method identifies regions of interest (ROIs) that show significant differences in response to stimuli based on statistical tests and a response magnitude threshold. The method takes two inputs: `p_values` and `magnitude_over_medians`.

The method first identifies significant ROIs based on the Kruskal-Wallis test, and then identifies ROIs with a response magnitude above the `response_magnitude_threshold` specified in the `config` attribute. If the `consider_only_positive` option is set to True in the `config` attribute, the method also requires ROIs to show a significant positive response according to the Wilcoxon signed rank test.

The method returns a set of indices of the significant ROIs in both the Kruskal-Wallis and magnitude datasets.

## Random Data Generation

Random data is generated using NumPy's `poisson` function. For each seed, the function generates a 3D array with dimensions `(n_sessions, n_roi, len_session)` that represents the fluorescence signal in two-photon microscopy. The values in the array are integers drawn from a Poisson distribution with an expected value that is randomly generated between 0 and 50 for each seed.

After generating the Poisson-distributed data, the sign of each value is randomly flipped to create a more diverse set of data.

## Proportions of Significant Results

|                           | Proportion                            |
|:--------------------------|:--------------------------------------|
| Kruskal-Wallis test       | <span style='color:red'>9.50%</span>  |
| Sign test                 | <span style='color:red'>40.50%</span> |
| Wilcoxon signed rank test | <span style='color:red'>48.00%</span> |
| response magnitude        | <span style='color:red'>81.50%</span> |
| significant_rois_positive | 0.50%                                 |
| significant_rois          | <span style='color:red'>9.50%</span>  |
## Count of Significant Results

|    |   Kruskal-Wallis test |   Sign test |   Wilcoxon signed rank test |   response magnitude |   significant_rois_positive |   significant_rois |
|---:|----------------------:|------------:|----------------------------:|---------------------:|----------------------------:|-------------------:|
|  0 |                     0 |           3 |                           4 |                    5 |                           0 |                  0 |
|  1 |                     0 |           2 |                           4 |                    5 |                           0 |                  0 |
|  2 |                     0 |           2 |                           0 |                    4 |                           0 |                  0 |
|  3 |                     0 |           1 |                           3 |                    4 |                           0 |                  0 |
|  4 |                     0 |           4 |                           5 |                    5 |                           0 |                  0 |
|  5 |                     0 |           0 |                           1 |                    2 |                           0 |                  0 |
|  6 |                     1 |           4 |                           4 |                    5 |                           1 |                  1 |
|  7 |                     1 |           3 |                           4 |                    5 |                           0 |                  1 |
|  8 |                     2 |           1 |                           2 |                    4 |                           0 |                  2 |
|  9 |                     0 |           4 |                           4 |                    3 |                           0 |                  0 |
| 10 |                     1 |           2 |                           2 |                    4 |                           0 |                  1 |
| 11 |                     1 |           2 |                           3 |                    4 |                           0 |                  1 |
| 12 |                     0 |           3 |                           3 |                    5 |                           0 |                  0 |
| 13 |                     0 |           2 |                           2 |                    5 |                           0 |                  0 |
| 14 |                     1 |           2 |                           3 |                    3 |                           0 |                  1 |
| 15 |                     0 |           3 |                           3 |                    3 |                           0 |                  0 |
| 16 |                     1 |           1 |                           0 |                    2 |                           0 |                  1 |
| 17 |                     0 |           2 |                           3 |                    3 |                           0 |                  0 |
| 18 |                     1 |           2 |                           2 |                    4 |                           0 |                  1 |
| 19 |                     0 |           1 |                           1 |                    4 |                           0 |                  0 |
| 20 |                     1 |           2 |                           3 |                    5 |                           0 |                  1 |
| 21 |                     0 |           3 |                           3 |                    5 |                           0 |                  0 |
| 22 |                     0 |           1 |                           2 |                    5 |                           0 |                  0 |
| 23 |                     0 |           3 |                           3 |                    4 |                           0 |                  0 |
| 24 |                     0 |           3 |                           3 |                    4 |                           0 |                  0 |
| 25 |                     0 |           0 |                           3 |                    4 |                           0 |                  0 |
| 26 |                     0 |           3 |                           3 |                    5 |                           0 |                  0 |
| 27 |                     1 |           2 |                           1 |                    3 |                           0 |                  1 |
| 28 |                     2 |           2 |                           2 |                    5 |                           0 |                  2 |
| 29 |                     0 |           2 |                           2 |                    3 |                           0 |                  0 |
| 30 |                     1 |           0 |                           1 |                    5 |                           0 |                  1 |
| 31 |                     0 |           1 |                           0 |                    4 |                           0 |                  0 |
| 32 |                     0 |           3 |                           3 |                    4 |                           0 |                  0 |
| 33 |                     0 |           2 |                           3 |                    5 |                           0 |                  0 |
| 34 |                     2 |           4 |                           3 |                    5 |                           0 |                  2 |
| 35 |                     1 |           0 |                           1 |                    4 |                           0 |                  1 |
| 36 |                     0 |           1 |                           2 |                    4 |                           0 |                  0 |
| 37 |                     1 |           1 |                           2 |                    3 |                           0 |                  1 |
| 38 |                     1 |           1 |                           1 |                    2 |                           0 |                  1 |
| 39 |                     0 |           3 |                           2 |                    5 |                           0 |                  0 |

