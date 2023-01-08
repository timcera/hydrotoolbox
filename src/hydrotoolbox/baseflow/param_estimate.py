import numpy as np

from .utils import NSE, moving_average, multi_arange


def recession_coefficient(Q, strict, date=None, ice_period=None):
    if ice_period is None:
        idx_ice = np.full(Q.shape, False)
    else:
        beg, end = ice_period
        if (end[0] > beg[0]) or ((end[0] == beg[0]) & (end[1] > beg[1])):
            idx_ice = (
                (date.M > beg[0]) & (date.M < end[0])
                | ((date.M == beg[0]) & (date.D >= beg[1]))
                | ((date.M == end[0]) & (date.D <= end[1]))
            )
        else:
            idx_ice = (
                (date.M > beg[0])
                | (date.M < end[0])
                | ((date.M == beg[0]) & (date.D >= beg[1]))
                | ((date.M == end[0]) & (date.D <= end[1]))
            )
    dry = strict[~idx_ice]

    cQ = Q[1:-1][dry[1:-1]]
    dQ = ((Q[2:] - Q[:-2]) / 2)[dry[1:-1]]

    idx = np.argsort(-dQ / cQ)[np.floor(dQ.shape[0] * 0.05).astype(int)]
    K = -cQ[idx] / dQ[idx]
    return np.exp(-1 / K)


def param_calibrate(param_range, method, Q, b_LH):
    idx_rec = recession_period(Q)
    idx_oth = np.full(Q.shape[0], True)
    idx_oth[idx_rec] = False
    return param_calibrate_jit(param_range, method, Q, b_LH, idx_rec, idx_oth)


def param_calibrate_jit(param_range, method, Q, b_LH, idx_rec, idx_oth):
    log_q = np.log(Q + 1)
    loss = np.zeros(param_range.shape)
    for i in range(param_range.shape[0]):
        p = param_range[i]
        b_exceed = method(Q, b_LH, p, return_exceed=True)
        f_exd, logb = b_exceed[-1] / Q.shape[0], np.log(b_exceed[:-1] + 1)
        NSE_rec = NSE(log_q[idx_rec], logb[idx_rec])
        NSE_oth = NSE(log_q[idx_oth], logb[idx_oth])
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


def maxmium_BFI(Q, b_LH, a, date=None):
    b = Backward(Q, b_LH, a)

    if date is None:
        idx_end = b.shape[0] // 365 * 365
        annual_b = np.mean(b[:idx_end].reshape(-1, 365), axis=1)
        annual_q = np.mean(Q[:idx_end].reshape(-1, 365), axis=1)
    else:
        idx_year = date.Y - date.Y.min()
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
        if b[i - 1] > Q[i - 1]:
            b[i - 1] = Q[i - 1]
    return b
