import numpy as np


def furey(Q, b_LH, k, c3c1):
    """Furey digital filter (Furey & Gupta, 2001, 2003)

    Args:
        Q (np.array): streamflow
        k (float): recession coefficient
        A (float): calibrated in baseflow.param_estimate
    """
    b = np.zeros(Q.shape[0])

    b[0] = b_LH[0]

    coefficient = (1 - k) * c3c1

    num_exceed = 0
    for i in range(Q.shape[0] - 1):
        b[i + 1] = k * b[i] + coefficient * (Q[i] - b[i])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    return b, num_exceed


def f_Furey(a):
    def _Furey(Q, b_LH, A):
        return furey(Q, b_LH, a, A)

    return _Furey
