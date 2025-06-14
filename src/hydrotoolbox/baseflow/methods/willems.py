import numpy as np


def willems(Q, b_LH, k, w):
    """Digital filter (Willems, 2009)"""
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]

    v = (1 - w) * (1 - k) / (2 * w)
    alpha = (k - v) / (1 + v)
    beta = v / (1 + v)

    num_exceed = 0
    for i in range(Q.shape[0] - 1):
        b[i + 1] = alpha * b[i] + beta * (Q[i] + Q[i + 1])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    return b, num_exceed


def f_Willems(a):
    def _Willems(Q, b_LH, w):
        return willems(Q, b_LH, a, w)

    return _Willems
