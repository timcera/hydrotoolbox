import numpy as np

from .local import hysep_interval


def slide(Q, area=None, num_days=None):
    """Slide interval graphical method from HYSEP program (Sloto & Crouse, 1996)

    Args:
        Q (np.array): streamflow
        area (float): basin area in km^2
    """
    inN = hysep_interval(area=area, num_days=num_days)
    return slide_interpolation(Q, inN)


def slide_interpolation(Q, inN):
    b = np.zeros(Q.shape[0])
    for i in range(np.int64((inN - 1) / 2), np.int64(Q.shape[0] - (inN - 1) / 2)):
        b[i] = np.min(Q[np.int64(i - (inN - 1) / 2) : np.int64(i + (inN + 1) / 2)])
    b[: np.int64((inN - 1) / 2)] = np.min(Q[: np.int64((inN - 1) / 2)])
    b[np.int64(Q.shape[0] - (inN - 1) / 2) :] = np.min(
        Q[np.int64(Q.shape[0] - (inN - 1) / 2) :]
    )
    return b
