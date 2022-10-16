import numpy as np


def Chapman(Q, b_LH, a, return_exceed=False):
    """Chapman filter (Chapman, 1991)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
    """
    b = np.zeros(Q.shape[0] + 1) if return_exceed else np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    first_c = (3 * a - 1) / (3 - a)
    second_c = (1 - a) / (3 - a)
    for i in range(Q.shape[0] - 1):
        b[i + 1] = first_c * b[i] + second_c * (Q[i + 1] + Q[i])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b
