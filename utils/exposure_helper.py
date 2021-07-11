import numpy as np
# Helper functions from https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition

def sRGBToLinear(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def linearTosRGB(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0031308, 1.055 * np.power(x, 1.0 / 2.4) - 0.055, x * 12.92)


def calculate_ev100_from_metadata(aperture_f: float, shutter_s: float, iso: int):
    ev_s = np.log2((aperture_f * aperture_f) / shutter_s)
    ev_100 = ev_s - np.log2(iso / 100)
    return ev_100


def calculate_luminance_from_ev100(ev100, q=0.65, S=100):
    return (78 / (q * S)) * np.power(2.0, ev100)


def convert_luminance(x: np.ndarray) -> np.ndarray:
    return 0.212671 * x[..., 0] + 0.71516 * x[..., 1] + 0.072169 * x[..., 2]


def smoothStep(x, edge0=0.0, edge1=1.0):
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def compute_avg_luminance(x: np.ndarray) -> np.ndarray:
    L = np.nan_to_num(convert_luminance(x))
    L = L * center_weight(L)

    if len(x.shape) == 3:
        axis = (0, 1)
    elif len(x.shape) == 4:
        axis = (1, 2)
    else:
        raise ValueError(
            "Only 3 dimensional (HWC) or 4 dimensionals (NHWC) images are supported"
        )
    avgL1 = np.average(L, axis=axis)
    return avgL1


def compute_ev100_from_avg_luminance(avgL, S=100.0, K=12.5):
    return np.log2(avgL * S / K)  # or 12.7


def convert_ev100_to_exp(ev100, q=0.65, S=100):
    maxL = (78 / (q * S)) * np.power(2.0, ev100)
    return np.clip(1.0 / maxL, 1e-7, None)


def compute_auto_exp(
    x: np.ndarray, clip: bool = True, returnEv100: bool = True
) -> np.ndarray:
    avgL = np.clip(compute_avg_luminance(x), 1e-5, None)
    ev100 = compute_ev100_from_avg_luminance(avgL)

    ret = apply_ev100(x, ev100, clip)
    if returnEv100:
        return ret, ev100
    else:
        return ret


def apply_ev100(x: np.ndarray, ev100, clip: bool = True):
    exp = convert_ev100_to_exp(ev100)  # This can become an invalid number. why?

    if len(x.shape) == 3:
        exposed = x * exp
    else:
        exposed = x * exp.reshape((exp.shape[0], *[1 for _ in x.shape[1:]]))
    if clip:
        exposed = np.clip(exposed, 0.0, 1.0)

    return exposed


def center_weight(x):
    def smoothStep(x, edge0=0.0, edge1=1.0):
        x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return x * x * x * (x * (x * 6 - 15) + 10)

    idx = np.argwhere(np.ones_like(x))
    idxs = np.reshape(idx, (*x.shape, len(x.shape)))

    if len(x.shape) == 2:
        axis = (0, 1)
    elif len(x.shape) == 3:
        axis = (1, 2)
        idxs = idxs[..., 1:]
    else:
        raise ValueError(
            "Only 2 dimensional (HW) or 3 dimensionals (NHW) images are supported"
        )

    center_idx = np.array([x.shape[axis[0]] / 2, x.shape[axis[1]] / 2])
    center_dist = np.linalg.norm(idxs - center_idx, axis=-1)

    return 1 - smoothStep(center_dist / x.shape[axis[1]] * 2)
