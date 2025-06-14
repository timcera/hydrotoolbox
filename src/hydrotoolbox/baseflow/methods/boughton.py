import numpy as np


def boughton(Q, b_LH, k, C):
    """Boughton double-parameter filter (Boughton, 2004)"""
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]

    alpha = k / (1 + C)
    beta = C / (1 + C)

    num_exceed = 0
    for i in range(Q.shape[0] - 1):
        b[i + 1] = alpha * b[i] + beta * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    return b, num_exceed


def f_Boughton(k):
    def _Boughton(Q, b_LH, C):
        return boughton(Q, b_LH, k, C)

    return _Boughton
