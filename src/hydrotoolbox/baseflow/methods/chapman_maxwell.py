__all__ = [
    "chapman_maxwell",
]

# First party imports
from hydrotoolbox.baseflow.general_form import general_form_digital_filter
from hydrotoolbox.baseflow.param_estimate import resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod


def chapman_maxwell(flow, k=None, b0_method: BaseflowInitialMethod = "zi"):
    """Chapman-Maxwell filter (Chapman & Maxwell, 1996).

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
    alpha = k / (2 - k)
    beta = (1 - k) / (2 - k)
    gamma = 0
    return general_form_digital_filter(
        flow, alpha, beta, gamma, b0_method=b0_method, k=k
    )[0]
