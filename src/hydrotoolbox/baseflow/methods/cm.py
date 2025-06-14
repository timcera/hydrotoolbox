import numpy as np


def cm(Q, b_LH, k):
    """CM filter (Chapman & Maxwell, 1996)

    Args:
        Q (np.array): streamflow
        k (float): recession coefficient
    """
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]

    alpha = k / (2 - k)
    beta = (1 - k) / (2 - k)

    num_exceed = 0
    for i in range(Q.shape[0] - 1):
        b[i + 1] = alpha * b[i] + beta * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    return b, num_exceed
