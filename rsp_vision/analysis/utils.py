from ..objects.enums import PhotonType


def ascii_array_to_string(array):
    return "".join([chr(int(i)) for i in array])


def get_fps(photon_type):
    if photon_type == PhotonType.TWO_PHOTON:
        return 30
    elif photon_type == PhotonType.THREE_PHOTON:
        return 15
    else:
        raise NotImplementedError(
            "Unknown number of frames per second for this photon type"
        )
