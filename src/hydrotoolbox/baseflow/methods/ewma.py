import numpy as np


def ewma(Q, b_LH, e):
    """exponential weighted moving average (EWMA) filter (Tularam & Ilahee, 2008)

    Args:
        Q (np.array): streamflow
        e (float): smoothing parameter
    """
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]

    first_c = 1 - e

    num_exceed = 0
    for i in range(Q.shape[0] - 1):
        b[i + 1] = first_c * b[i] + e * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    return b, num_exceed
