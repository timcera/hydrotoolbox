__all__ = [
    "willems",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.general_form import general_form_digital_filter
from hydrotoolbox.baseflow.param_estimate import param_calibrate, resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod


def _willems(flow, k, w, b0_method: BaseflowInitialMethod = "zi"):
    v = (1 - w) * (1 - k) / (2 * w)
    alpha = (k - v) / (1 + v)
    beta = v / (1 + v)
    gamma = 1
    return general_form_digital_filter(
        flow, alpha, beta, gamma, b0_method=b0_method, k=k
    )


def willems(flow, k=None, w=None, b0_method: BaseflowInitialMethod = "zi"):
    """Digital filter (Willems, 2009).

    Parameters
    ----------
    flow : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient, estimated if None.
    w : float, optional
        Filter parameter, calibrated if None.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or value. "zi" uses SciPy filter
        initialization. Numeric strings are also accepted.

    Returns
    -------
    ndarray
        Separated baseflow.
    """
    k = resolve_k(flow, k)
    if w is None:
        w = param_calibrate(
            np.arange(0.001, 1, 0.001), _willems, flow, k, b0_method=b0_method
        )
    return _willems(flow, k, float(w), b0_method=b0_method)[0]
