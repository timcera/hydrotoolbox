__all__ = [
    "eckhardt",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.general_form import general_form_digital_filter
from hydrotoolbox.baseflow.param_estimate import param_calibrate, resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod


def _eckhardt(flow, k, BFImax, b0_method: BaseflowInitialMethod = "zi"):
    denom = 1 - k * BFImax
    alpha = ((1 - BFImax) * k) / denom
    beta = ((1 - k) * BFImax) / denom
    gamma = 0
    return general_form_digital_filter(
        flow, alpha, beta, gamma, b0_method=b0_method, k=k
    )


def eckhardt(flow, k=None, bfi_max=None, b0_method: BaseflowInitialMethod = "zi"):
    """Eckhardt filter (Eckhardt, 2005).

    Parameters
    ----------
    flow : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient, estimated if None.
    bfi_max : float, optional
        Maximum baseflow index, calibrated if None.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or value. "zi" uses SciPy filter
        initialization. Numeric strings are also accepted.

    Returns
    -------
    ndarray
        Separated baseflow.
    """
    k = resolve_k(flow, k)
    if bfi_max is None:
        bfi_max = param_calibrate(
            np.arange(0.001, 1, 0.001), _eckhardt, flow, k, b0_method=b0_method
        )
    return _eckhardt(flow, k, float(bfi_max), b0_method=b0_method)[0]
