__all__ = [
    "furey",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.param_estimate import param_calibrate, resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod, _init_baseflow


def _furey(flow, k, c3c1, b0_method: BaseflowInitialMethod = "Q0"):
    beta = c3c1 * (1 - k)
    alpha = k - beta
    b = np.zeros(flow.shape[0])

    b[0] = _init_baseflow(
        flow, b0_method, k=k, alpha=alpha, beta=beta, gamma=1, delta=0
    )

    num_exceed = 0
    for i in range(1, flow.shape[0]):
        b[i] = alpha * b[i - 1] + beta * flow[i - 1]
        if b[i] > flow[i]:
            b[i] = flow[i]
            num_exceed += 1

    return b, num_exceed


def furey(flow, k=None, c3c1=None, b0_method: BaseflowInitialMethod = "Q0"):
    """Furey digital filter (Furey & Gupta, 2001, 2003).

    Parameters
    ----------
    flow : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient, estimated if None.
    c3c1 : float, optional
        Filter parameter, calibrated if None.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or value. "zi" uses SciPy filter
        initialization. Numeric strings are accepted. Defaults to "Q0".

    Returns
    -------
    ndarray
        Separated baseflow.
    """
    k = resolve_k(flow, k)
    if c3c1 is None:
        c3c1 = param_calibrate(
            np.arange(0.01, 10, 0.01), _furey, flow, k, b0_method=b0_method
        )

    return _furey(flow, k, float(c3c1), b0_method)[0]
