import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from plotly.subplots import make_subplots

path = Path(
    "/Users/lauraporta/local_data/rsp_vision/derivatives/sub-003_line-CX_id-1112654/ses-000_hemisphere-hL_region-RSPd_monitor-front"
)

with open(path / "gaussians_fits_and_roi_info.pickle", "rb") as f:
    data = pickle.load(f)

median_subtracted_response = data["median_subtracted_responses"]
n_roi = data["n_roi"]
responsive_roi = data["responsive_rois"]
fit_outputs = data["fit_outputs"]

#  scatterplot between sigma tf and sigma sf, hue is power law exponent

sigma_sf = []
sigma_tf = []
𝜻_power_law_exp = []

for roi in range(n_roi):
    if roi in responsive_roi:
        key = (roi, "pooled")
        _, _, _, sig_sf, sig_tf, 𝜻 = fit_outputs[key]
        if 𝜻 > 0:
            sigma_sf.append(sig_sf)
            sigma_tf.append(sig_tf)
            𝜻_power_law_exp.append(𝜻)

fig = make_subplots(rows=1, cols=2)

#  color red if
fig.add_trace(
    go.Scatter(
        x=sigma_sf,
        y=sigma_tf,
        mode="markers",
        marker=dict(
            color=𝜻_power_law_exp,
            size=5,
        ),
        name="Response",
    ),
    row=1,
    col=1,
)

fig.update_layout(
    plot_bgcolor="white",
)


sfs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
tfs = [0.5, 1, 2, 4, 8, 16]

velocity = np.zeros((6, 6))
for i, sf in enumerate(sfs):
    for j, tf in enumerate(tfs):
        velocity[i, j] = sf / tf

flat_velocity = velocity.flatten()
flat_velocity = np.round(flat_velocity, 4)

for roi in range(n_roi):
    key = (roi, "pooled")
    msr = median_subtracted_response[key]
    flat_msr = msr.flatten()
    unique_velocity = np.unique(flat_velocity)

    max_response_per_velocity = []
    for vel in unique_velocity:
        max_response_per_velocity.append(
            np.max(flat_msr[flat_velocity == vel])
        )

    fig.add_trace(
        go.Scatter(
            x=unique_velocity,
            y=max_response_per_velocity,
            mode="lines",
            marker=dict(
                color="red" if roi in responsive_roi else "lightblue",
                size=5,
            ),
            name=f"ROI {roi + 1}",
        ),
        row=1,
        col=2,
    )

fig.update_xaxes(type="log", title_text="Speed deg/s", row=1, col=2)
fig.update_yaxes(title_text="Response ΔF/F", row=1, col=2)


app = Dash()
app.layout = html.Div(
    [
        dcc.Graph(figure=fig),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8052)
