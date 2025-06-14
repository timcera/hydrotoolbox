import numpy as np


def lh(Q, alpha=0.925):
    """LH digital filter (Lyne & Hollick, 1979)"""
    b = np.zeros(Q.shape[0])

    first_c = (1 - alpha) / 2

    num_exceed = 0
    # first pass
    b[0] = Q[0] / 2
    for i in range(Q.shape[0] - 1):
        b[i + 1] = alpha * b[i] + first_c * (Q[i] + Q[i + 1])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    # second pass
    b1 = np.copy(b)
    for i in range(Q.shape[0] - 2, -1, -1):
        b[i] = alpha * b[i + 1] + first_c * (b1[i + 1] + b1[i])
        if b[i] > Q[i]:
            b[i] = Q[i]
            num_exceed += 1

    return b, num_exceed
