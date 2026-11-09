# Standard library imports

__all__ = [
    "usgs_hysep_local",
]

# Standard library imports
import math

# Third party imports
import numpy as np
from pint import UnitRegistry
from pydantic import validate_call

# First party imports
from hydrotoolbox.baseflow.methods.lyne_hollick import lyne_hollick
from hydrotoolbox.baseflow.methods.ukih import linear_interpolation
from hydrotoolbox.baseflow.param_estimate import resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod

ureg = UnitRegistry()


def usgs_hysep_local(
    Q, k=None, area=None, num_days=None, b0_method: BaseflowInitialMethod = "zi"
):
    """Local minimum graphical method from HYSEP (Sloto & Crouse, 1996).

    Parameters
    ----------
    Q : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient used for the Lyne-Hollick end values.
        Estimated from the streamflow if None.
    area : float, optional
        Basin area in square kilometers.
    num_days : int, optional
        Override the interval calculated from the basin area.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow for the Lyne-Hollick series used at the edges.
        "zi" uses SciPy filter initialization. Numeric strings are accepted.

    Returns
    -------
    ndarray
        Separated baseflow.
    """
    b_LH = lyne_hollick(Q, resolve_k(Q, k), b0_method=b0_method)
    idx_turn = local_turn(Q, hysep_interval(area=area, num_days=num_days))
    b = linear_interpolation(Q, idx_turn)[0]
    b[: idx_turn[0]] = b_LH[: idx_turn[0]]
    b[idx_turn[-1] + 1 :] = b_LH[idx_turn[-1] + 1 :]
    return b


@validate_call
def hysep_interval(
    area: float | None = None, area_unit: str = "mile**2", num_days=None
) -> int:
    """
    Estimate the duration of surface runoff.

    Is calculated from the empirical relation: N=A^0.2, (1) where N is the
    number of days after which surface runoff ceases, and A is the drainage
    area in square miles (Linsley and others, 1982, p. 210).  The interval 2N*
    used for hydrograph separations is the odd integer between 3 and 11 nearest
    to 2N (Pettyjohn and Henning, 1979, p. 31).

    Parameters
    ----------
    area : float | None, optional
        Basin area.
    area_unit : str, optional
        Unit of the basin area. Default is "mile**2".
    num_days : int, optional
        Override the interval calculated from the basin area.

    Returns
    -------
    int
        Estimated HYSEP interval.
    """
    if area is None and num_days is None:
        num_days = 5
    elif area is not None:
        area = area * ureg(area_unit)
        area = area.to("miles**2").magnitude
        num_days = area**0.2
    inn = math.ceil(2 * float(num_days))
    if (inn % 2) == 0:
        inn = inn - 1
    inn = int(min(max(inn, 3), 11))
    return inn


def local_turn(Q, inN):
    idx_turn = np.zeros(Q.shape[0], dtype=np.int64)
    for i in range(np.int64((inN - 1) / 2), np.int64(Q.shape[0] - (inN - 1) / 2)):
        if Q[i] == np.min(Q[np.int64(i - (inN - 1) / 2) : np.int64(i + (inN + 1) / 2)]):
            idx_turn[i] = i
    return idx_turn[idx_turn != 0]
