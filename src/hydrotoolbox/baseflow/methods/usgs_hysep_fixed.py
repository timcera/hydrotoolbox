__all__ = [
    "usgs_hysep_fixed",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.methods.usgs_hysep_local import hysep_interval


def usgs_hysep_fixed(Q, area=None, num_days=None):
    """Fixed interval graphical method from HYSEP (Sloto & Crouse, 1996).

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
    return fixed_interpolation(Q, inN)


def fixed_interpolation(Q, inN):
    b = np.zeros(Q.shape[0])
    n = Q.shape[0] // inN
    for i in range(n):
        b[inN * i : inN * (i + 1)] = np.min(Q[inN * i : inN * (i + 1)])
    if n * inN != Q.shape[0]:
        b[n * inN :] = np.min(Q[n * inN :])
    mask = b > Q
    b[mask] = Q[mask]
    return b
