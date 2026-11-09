__all__ = [
    "boughton",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.general_form import general_form_digital_filter
from hydrotoolbox.baseflow.param_estimate import param_calibrate, resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod


def _boughton(flow, k, C, b0_method: BaseflowInitialMethod = "zi"):
    alpha = k / (1 + C)
    beta = C / (1 + C)
    gamma = 0.0
    return general_form_digital_filter(
        flow, alpha, beta, gamma, b0_method=b0_method, k=k
    )


def boughton(flow, k=None, C=None, b0_method: BaseflowInitialMethod = "zi"):
    """Boughton double-parameter filter (Boughton, 2004).

    Parameters
    ----------
    flow : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient, estimated if None.
    C : float, optional
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
    if C is None:
        C = param_calibrate(
            np.arange(0.001, 1, 0.001), _boughton, flow, k, b0_method=b0_method
        )
    return _boughton(flow, k, float(C), b0_method=b0_method)[0]
