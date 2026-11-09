# Third party imports
import numpy as np

# First party imports
from hydrotoolbox.baseflow.comparison import strict_baseflow
from hydrotoolbox.baseflow.utils import (
    NSE,
    BaseflowInitialMethod,
    _init_baseflow,
    moving_average,
    multi_arange,
)


def recession_coefficient(Q, strict):
    cQ, dQ = Q[1:-1], (Q[2:] - Q[:-2]) / 2
    cQ, dQ = cQ[strict[1:-1]], dQ[strict[1:-1]]

    idx = np.argsort(-dQ / cQ)[np.floor(dQ.shape[0] * 0.05).astype(int)]
    K = -cQ[idx] / dQ[idx]
    return np.exp(-1 / K)


def resolve_k(flow, k):
    """Return k, estimating the recession coefficient from flow if None."""
    if k is None:
        return float(recession_coefficient(flow, strict_baseflow(flow)))
    return float(k)


def param_calibrate(param_range, method, Q, k, b0_method: BaseflowInitialMethod = "Q0"):
    """
    Calibrate the parameter for baseflow separation using the specified method.

    1. Identify recession periods using 'recession_period(Q)'.

        'recession_period(Q)' finds stretches where the 3-day average flow keeps
        falling for at least 10 days, and keeps the last 40% of each stretch.
        Those are the days where flow is almost entirely groundwater.
        Everything else (idx_other) is storm-influenced.

    2. Score each candidate parameter p. For each p, it runs the filter and compares baseflow b with total flow Q on a log scale. Log scale weights low flows, which is where baseflow lives.

        NSE_rec: how well b matches Q on recession days. Baseflow should equal
        flow there, so higher is better.

        NSE_oth: how well b matches Q on other days. Baseflow should be well
        below flow there, so a good filter gets a low NSE.

        f_exd: the fraction of days where the filter produced b > Q and had to
        be capped. This is physically impossible, so the fraction should be
        small.

    3. Combine the three scores into one loss:

        loss = 1 - (1 - (1-NSE_rec)/(1-NSE_oth)) * (1 - f_exd)

        The loss is small when NSE_rec is high, NSE_oth is low, and few days
        exceed flow. It then returns the p with the lowest loss.

    Parameters
    ----------
    param_range : ndarray
        Array of parameter values to test.
    method : callable
        Function to calculate baseflow and the number of exceedances.
    Q : ndarray
        Streamflow time series.
    k : float
        Additional parameter for the method.
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow method or value. "zi" is calculated for each
        candidate's filter coefficients. Numeric strings are also accepted.

    Returns
    -------
    float
        Candidate parameter value minimizing the loss function.
    """
    idx_recession = recession_period(Q)
    idx_other = np.full(Q.shape[0], True)
    idx_other[idx_recession] = False
    logQ = np.log1p(Q)
    logQ_idx_recession = logQ[idx_recession]
    logQ_idx_other = logQ[idx_other]
    loss = np.zeros(param_range.shape)
    if b0_method != "zi":
        b0_method = _init_baseflow(Q, b0_method, k=k)
    for i in range(param_range.shape[0]):
        p = param_range[i]
        b_exceed, num_exceed = method(Q, k, p, b0_method=b0_method)
        f_exd = num_exceed / Q.shape[0]
        logb = np.log1p(b_exceed)
        NSE_rec = NSE(logQ_idx_recession, logb[idx_recession])
        NSE_oth = NSE(logQ_idx_other, logb[idx_other])
        loss[i] = 1 - (1 - (1 - NSE_rec) / (1 - NSE_oth)) * (1 - f_exd)
    return param_range[np.argmin(loss)]


def recession_period(Q):
    idx_dec = np.zeros(Q.shape[0] - 1, dtype=np.int64)
    q_ave = moving_average(Q, 3)
    idx_dec[1:-1] = (q_ave[:-1] - q_ave[1:]) > 0
    idx_beg = np.where(idx_dec[:-1] - idx_dec[1:] == -1)[0] + 1
    idx_end = np.where(idx_dec[:-1] - idx_dec[1:] == 1)[0] + 1
    idx_keep = (idx_end - idx_beg) >= 10
    idx_beg = idx_beg[idx_keep]
    idx_end = idx_end[idx_keep]
    duration = idx_end - idx_beg
    idx_beg = idx_beg + np.ceil(duration * 0.6).astype(np.int64)
    return multi_arange(idx_beg, idx_end)


def maximum_BFI(Q, b_LH, a, date=None):
    b = Backward(Q, b_LH, a)

    if date is None:
        idx_end = b.shape[0] // 365 * 365
        annual_b = np.mean(b[:idx_end].reshape(-1, 365), axis=1)
        annual_q = np.mean(Q[:idx_end].reshape(-1, 365), axis=1)
    else:
        idx_year = date.year - date.year.min()
        counts = np.bincount(idx_year)
        idx_valid = counts > 0
        annual_b = np.bincount(idx_year, weights=b)[idx_valid] / counts[idx_valid]
        annual_q = np.bincount(idx_year, weights=Q)[idx_valid] / counts[idx_valid]
    annual_bfi = annual_b / annual_q
    bfi_max = np.max(annual_bfi)
    bfi_max = bfi_max if bfi_max < 0.9 else np.sum(annual_b) / np.sum(annual_q)
    return bfi_max


def Backward(Q, b_LH, a):
    b = np.zeros(Q.shape[0])
    b[-1] = b_LH[-1]
    for i in range(Q.shape[0] - 1, 0, -1):
        b[i - 1] = b[i] / a
        if b[i] == 0:
            b[i - 1] = Q[i - 1]
        b[i - 1] = min(b[i - 1], Q[i - 1])
    return b
