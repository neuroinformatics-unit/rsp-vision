from ..objects.enums import PhotonType


def get_fps(photon_type, config: dict):
    if photon_type == PhotonType.TWO_PHOTON:
        return config["fps_two_photon"]
    elif photon_type == PhotonType.THREE_PHOTON:
        return config["fps_three_photon"]
    else:
        raise NotImplementedError(
            "Unknown number of frames per second for this photon type"
        )
