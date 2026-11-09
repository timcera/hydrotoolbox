__all__ = [
    "ewma",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.param_estimate import param_calibrate, resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod, _init_baseflow


def _ewma(Q, k, e, b0_method: BaseflowInitialMethod = "LH"):
    """Exponential weighted moving average filter (Tularam & Ilahee, 2008).

    Parameters
    ----------
    Q : ndarray
        Streamflow.
    k : float
        Recession coefficient used for Lyne-Hollick initialization.
    e : float
        Smoothing parameter.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or explicit value. "zi" uses SciPy filter initialization.

    Returns
    -------
    b : ndarray
        Separated baseflow.
    num_exceed : int
        Number of baseflow values capped at streamflow.
    """
    b = np.zeros(Q.shape[0])
    b[0] = _init_baseflow(Q, b0_method, k=k, alpha=1 - e, beta=e, gamma=0)

    first_c = 1 - e

    num_exceed = 0
    for i in range(Q.shape[0] - 1):
        b[i + 1] = first_c * b[i] + e * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    return b, num_exceed


def ewma(Q, k=None, e=None, b0_method: BaseflowInitialMethod = "LH"):
    """Exponential weighted moving average filter (Tularam & Ilahee, 2008).

    Parameters
    ----------
    Q : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient used for the Lyne-Hollick starting value.
        Estimated from the streamflow if None.
    e : float, optional
        Smoothing parameter, calibrated if None.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or explicit value. "zi" uses SciPy filter
        initialization. Numeric strings are accepted. Defaults to "LH".

    Returns
    -------
    ndarray
        Separated baseflow.
    """
    k = resolve_k(Q, k)
    if e is None:
        e = param_calibrate(
            np.arange(0.001, 0.5, 0.001), _ewma, Q, k, b0_method=b0_method
        )
    return _ewma(Q, k, float(e), b0_method=b0_method)[0]
