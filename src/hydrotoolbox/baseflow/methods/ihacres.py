"""Tools for hydrology.

hydrotoolbox baseflow_sep --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow_sep sliding < daily.csv
hydrotoolbox baseflow_sep eckhardt,sliding < daily.csv
...

hydrotoolbox recession"""

__all__ = [
    "ihacres",
]

# Standard library imports
from functools import partial

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.general_form import general_form_digital_filter
from hydrotoolbox.baseflow.param_estimate import param_calibrate, resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod


def _ihacres(flow, k, C, a, b0_method: BaseflowInitialMethod = "zi"):
    alpha = k / (1 + C)
    beta = C / (1 + C)
    gamma = a
    return general_form_digital_filter(
        flow, alpha, beta, gamma, b0_method=b0_method, k=k
    )


def ihacres(flow, k=None, C=None, a=None, b0_method: BaseflowInitialMethod = "zi"):
    """IHACRES baseflow separation.

    Jakeman-Hornberger digital filter (Jakeman and Hornberger, 1993)

    Parameters
    ----------
    flow : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient, estimated if None.
    C : float, optional
        Filter parameter, calibrated if None.
    a : float, optional
        Previous-flow coefficient. None is treated as 0.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or value. "zi" uses SciPy filter
        initialization. Numeric strings are also accepted.

    Returns
    -------
    ndarray
        Separated baseflow.
    """
    k = resolve_k(flow, k)
    a = 0.0 if a is None else float(a)
    if C is None:
        C = param_calibrate(
            np.arange(0.001, 1, 0.001),
            partial(_ihacres, a=a),
            flow,
            k,
            b0_method=b0_method,
        )
    return _ihacres(flow, k, float(C), a, b0_method=b0_method)[0]
