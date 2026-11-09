__all__ = [
    "chapman",
]

# First party imports
from hydrotoolbox.baseflow.general_form import general_form_digital_filter
from hydrotoolbox.baseflow.param_estimate import resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod


def chapman(flow, k=None, b0_method: BaseflowInitialMethod = "zi"):
    """Chapman filter (Chapman, 1991).

    Parameters
    ----------
    flow : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient, estimated if None.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or value. "zi" uses SciPy filter
        initialization. Numeric strings are also accepted.

    Returns
    -------
    ndarray
        Separated baseflow.
    """
    k = resolve_k(flow, k)
    alpha = (3 * k - 1) / (3 - k)
    beta = (1 - k) / (3 - k)
    gamma = 1.0
    return general_form_digital_filter(
        flow, alpha, beta, gamma, b0_method=b0_method, k=k
    )[0]
