__all__ = [
    "ukih",
]

# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.methods.lyne_hollick import lyne_hollick
from hydrotoolbox.baseflow.param_estimate import resolve_k
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod


def ukih(Q, k=None, b0_method: BaseflowInitialMethod = "zi"):
    """Graphical method developed by the UK Institute of Hydrology (UKIH, 1980).

    Parameters
    ----------
    Q : ndarray
        Streamflow.
    k : float, optional
        Recession coefficient used for the Lyne-Hollick end values.
        Estimated from the streamflow if None.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow for the Lyne-Hollick series used at the edges.
        "zi" uses SciPy filter initialization. Numeric strings are accepted.

    Returns
    -------
    ndarray
        Separated baseflow.
    """
    b_LH = lyne_hollick(Q, resolve_k(Q, k), b0_method=b0_method)
    N = 5
    block_end = Q.shape[0] // N * N
    idx_min = np.argmin(Q[:block_end].reshape(-1, N), axis=1)
    idx_min = idx_min + np.arange(0, block_end, N)
    idx_turn = ukih_turn(Q, idx_min)
    b = linear_interpolation(Q, idx_turn)[0]
    b[: idx_turn[0]] = b_LH[: idx_turn[0]]
    b[idx_turn[-1] + 1 :] = b_LH[idx_turn[-1] + 1 :]
    return b


def ukih_turn(Q, idx_min):
    idx_turn = np.zeros(idx_min.shape[0], dtype=np.int64)
    for i in range(idx_min.shape[0] - 2):
        if (0.9 * Q[idx_min[i + 1]] < Q[idx_min[i]]) & (
            0.9 * Q[idx_min[i + 1]] < Q[idx_min[i + 2]]
        ):
            idx_turn[i] = idx_min[i + 1]
    return idx_turn[idx_turn != 0]


def linear_interpolation(Q, idx_turn):
    b = np.zeros(Q.shape[0])
    n = 0

    num_exceed = 0
    for i in range(idx_turn[0], idx_turn[-1] + 1):
        if i == idx_turn[n + 1]:
            n += 1
            b[i] = Q[i]
        else:
            b[i] = Q[idx_turn[n]] + (Q[idx_turn[n + 1]] - Q[idx_turn[n]]) / (
                idx_turn[n + 1] - idx_turn[n]
            ) * (i - idx_turn[n])
        if b[i] > Q[i]:
            b[i] = Q[i]
            num_exceed += 1

    return b, num_exceed
