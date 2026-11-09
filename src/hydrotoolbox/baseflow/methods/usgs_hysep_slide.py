__all__ = [
    "usgs_hysep_slide",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.methods.usgs_hysep_local import hysep_interval


def usgs_hysep_slide(Q, area=None, num_days=None):
    """Sliding interval graphical method from HYSEP (Sloto & Crouse, 1996).

    Parameters
    ----------
    Q : ndarray
        Streamflow.
    area : float, optional
        Basin area in square kilometers.
    num_days : int, optional
        Override the interval calculated from the basin area.

    Returns
    -------
    ndarray
        Separated baseflow.
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
