__all__ = [
    "lyne_hollick",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod, _init_baseflow


def lyne_hollick(flow, k, passes=2, b0_method: BaseflowInitialMethod = "zi"):
    """Lyne-Hollick digital filter (Lyne & Hollick, 1979).

    Parameters
    ----------
    flow : ndarray
        Streamflow.
    k : float
        Recession coefficient.
    passes : int, optional
        Number of filter passes.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or value, applied only to the first pass.
        "zi" uses the existing SciPy initialization. Use "Q0" to start at
        the first flow value. Numeric strings are also accepted.

    Returns
    -------
    ndarray
        Separated baseflow.

    Notes
    -----
    Each calculated value is bounded between zero and the current pass's flow
    before it is used in the next recurrence step. Each later pass starts at
    the previous pass's final value.
    """
    beta = (1 - k) / 2

    initial = _init_baseflow(flow, b0_method, k=k, alpha=k, beta=beta, gamma=1.0)

    baseflow = np.empty(len(flow), dtype=float)
    baseflow[0] = np.clip(initial, 0, flow[0])

    for pss in range(passes):
        if pss > 0:
            flow = baseflow[::-1]
            baseflow = flow.copy()
        for i in range(1, len(flow)):
            value = k * baseflow[i - 1] + beta * (flow[i - 1] + flow[i])
            baseflow[i] = min(max(value, 0), flow[i])

    if passes % 2 == 0:
        baseflow = baseflow[::-1]

    return baseflow
