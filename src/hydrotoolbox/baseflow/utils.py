# Standard library imports
from typing import Literal

# Third party imports
import numpy as np
from pint import UnitRegistry
from scipy.signal import lfilter_zi

ureg = UnitRegistry()

BaseflowInitialMethod = Literal["Q0", "min", "LH", "zi"] | str | float


def moving_average(x, w):
    res = np.convolve(x, np.ones(w)) / w
    return res[w - 1 : -w + 1]


def multi_arange_steps(starts, stops, steps):
    pos = 0
    cnt = np.sum((stops - starts + steps - np.sign(steps)) // steps, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)
    for i in range(starts.size):
        v, stop, step = starts[i], stops[i], steps[i]
        if step > 0:
            while v < stop:
                res[pos] = v
                pos += 1
                v += step
        elif step < 0:
            while v > stop:
                res[pos] = v
                pos += 1
                v += step
    assert pos == cnt
    return res


def multi_arange(starts, stops):
    pos = 0
    cnt = np.sum(stops - starts, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)
    for i in range(starts.size):
        num = stops[i] - starts[i]
        res[pos : pos + num] = np.arange(starts[i], stops[i])
        pos += num
    return res


def NSE(q_obs, q_sim):
    ss_res = np.sum(np.square(q_obs - q_sim))
    ss_tot = np.sum(np.square(q_obs - np.mean(q_obs)))
    return (1 - ss_res / (ss_tot + 1e-10)) - 1e-10


def _init_baseflow(
    Q,
    initial_method: BaseflowInitialMethod = "Q0",
    k=None,
    *,
    alpha=None,
    beta=None,
    gamma=None,
    delta=1,
):
    """Compute the initial baseflow value.

    Parameters
    ----------
    Q : ndarray
        Streamflow array.
    initial_method : {"Q0", "min", "LH", "zi"} or float
        Use the first flow, minimum flow, first Lyne-Hollick baseflow, SciPy
        filter initialization, or an explicit value. Numeric strings are accepted.
    k : float, optional
        Recession coefficient for the Lyne-Hollick initialization.
    alpha, beta, gamma : float, optional
        Filter coefficients, required for "zi".
    delta : float, optional
        Current-flow coefficient multiplier for "zi".

    Returns
    -------
    float
        Initial baseflow value.
    """
    if isinstance(initial_method, str):
        if initial_method == "Q0":
            return Q[0]
        elif initial_method == "min":
            return np.min(Q)
        elif initial_method == "LH":
            # First party imports
            from hydrotoolbox.baseflow.methods.lyne_hollick import lyne_hollick

            if k is None:
                # First party imports
                from hydrotoolbox.baseflow.param_estimate import resolve_k

                k = resolve_k(Q, k)
            return lyne_hollick(Q, k, b0_method="zi")[0]
        elif initial_method == "zi":
            if alpha is None or beta is None or gamma is None:
                raise ValueError('"zi" initialization requires alpha, beta and gamma')
            b = [beta * delta, beta * gamma]
            return lfilter_zi(b, [1.0, -alpha])[0] + b[0] * Q[0]
        else:
            try:
                return float(initial_method)
            except ValueError as exc:
                raise ValueError(f"Invalid initial_method: {initial_method}") from exc
    if initial_method is None:
        raise ValueError('initial_method cannot be None; use "zi" explicitly')
    return float(initial_method)
