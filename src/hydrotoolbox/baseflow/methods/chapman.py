import numpy as np


def chapman(Q, b_LH, k):
    """Chapman filter (Chapman, 1991)"""
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]

    c1 = (3 * k - 1) / (3 - k)
    c2 = (1 - k) / (3 - k)

    num_exceed = 0
    for i in range(Q.shape[0] - 1):
        b[i + 1] = c1 * b[i] + c2 * (Q[i + 1] + Q[i])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    return b, num_exceed
