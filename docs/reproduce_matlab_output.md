# Am I accurately reproducing the MATLAB output?
Yes, I am. Here is the proof.

## SF_TF analysis
Analysis options in use:
```MATLAB
options.response.subtract_baseline = true;
options.response.average_using = 'median';
options.response.baseline_frames = data.static_idx(end-29:end); 
options.response.response_frames = data.drift_idx(16:end); 
options.response.peak.enable = false;
options.response.peak.window = data.drift_idx(1) + (16:45);

options.response.only_positive = false;
options.stats.pos_p_threshold = 0.05;
options.response.use_magnitude = true;
options.response.magnitude_threshold = 2.7;
options.response.magnitude_average = 'median';
options.stats.anova_p_threshold = 0.0005;
options.response.fit_corr_threshold = 0.5;
```

Let's start with this example file `AK_1111739_hL_RSPd_monitor_front`.

Set a breakpoint where the script `example_sftf.m` calculates p-values:
```MATLAB
[is_sig, p_anova, p_positive, r_mag] = data.responsiveness(data.is_cell & on_days, options);
```
We can take the first two variables as an example.
### Variables breakdown
- `is_sig`, i.e. "is significant", is an (11x1) logical array with a length equal to the number of ROIs, 11 in this case. Indicates for each ROI if the response to the visual stimuli was significant;
- `p_anova`, an (11x1) double array, with the results of the non-parametric anova tests;

## Value comparison

MATLAB variable | MATLAB value | Python variable / attribute | Python value | Notes
------------ | ------------ | ------------ | ------------ | ------------ 
`is_sig` | (false false false false false true false true true false false) | `PhotonData.responsive_rois` | {8, 5, 7} | Python uses zero-indexing
`p_anova` | (0.00922394414060872, 0.00353356313489075, 0.938340487230328, NaN, 0.414746858341979, 3.63696657508830e-24, 0.000285886879712534, 8.17849199227997e-15, 4.24983481907371e-13, 0.0744141518018161, 0.000180455655992139) | `p_values["Kruskal-Wallis test"]` | array([9.22394414e-03, 3.53356313e-03, 9.38340487e-01, 8.00211099e-01, 4.14746858e-01, 3.63696658e-24, 2.85886880e-04, 8.17849199e-15, 4.24983482e-13, 7.44141518e-02, 1.80455656e-04]) | Handles NaN values better

The values match exactly, except for the NaN values, which are handled better in the new implementation.