import pathlib
import pickle
import time

from generate_mock_data import get_response_mock
from tqdm import tqdm


def calculate_stats_outputs(seeds):
    outputs = {}
    for seed in tqdm(seeds):
        response = get_response_mock(seed)

        max_attempts = 20

        for _ in range(max_attempts):
            try:
                response()
                break
            except AttributeError:
                print("Failed to get a valid response. Trying again...")
                time.sleep(0.5)
                pass
        else:
            message = "Failed to get a valid response \
                after {max_attempts} attempts. Seed: {seed}"
            raise RuntimeError(message)

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

    path = pathlib.Path(__file__).parent.absolute()
    output_path = path / "mock_data" / "outputs.plk"
    with open(output_path, "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    #  seed 5 causes troubles, so we exclude it
    calculate_stats_outputs(seeds)
