import pathlib
import pickle

from tests.fixtures_helpers import get_response_mock
from tqdm import tqdm


def calculate_stats_outputs(seeds):
    outputs = {}
    last_seed = seeds[-1]
    seeds = seeds[:-1]
    for seed in tqdm(seeds):
        response = get_response_mock(seed)
        response() 
        data = response.data

        outputs[str(seed)] = {
            "responses": data.responses.to_dict(),
            "p_values": data.p_values,
            "magnitude_over_medians": data.magnitude_over_medians.to_dict(),
            "responsive_rois": data.responsive_rois,
            "measured_preference": data.measured_preference,
            "fit_output": data.fit_output,
            "median_subtracted_response": data.median_subtracted_response,
        }

    # For the last seed we simulate two days of data
    response = get_response_mock(last_seed, multiple_days=True)
    response()
    data = response.data
    outputs[str(last_seed)] = {
        "responses": data.responses.to_dict(),
        "p_values": data.p_values,
        "magnitude_over_medians": data.magnitude_over_medians.to_dict(),
        "responsive_rois": data.responsive_rois,
        "measured_preference": data.measured_preference,
        "fit_output": data.fit_output,
        "median_subtracted_response": data.median_subtracted_response,
    }

    path = pathlib.Path(__file__).parent.absolute()
    output_path = path / "mock_data" / "outputs.plk"
    with open(output_path, "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    #  seed 5 causes troubles, so we exclude it
    calculate_stats_outputs(seeds)
