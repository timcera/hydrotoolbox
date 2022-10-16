import numpy as np


def CM(Q, b_LH, a, return_exceed=False):
    """CM filter (Chapman & Maxwell, 1996)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
    """
    b = np.zeros(Q.shape[0] + 1) if return_exceed else np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    first_c = a / (2 - 1)
    second_c = (1 - a) / (2 - a)
    for i in range(Q.shape[0] - 1):
        b[i + 1] = first_c * b[i] + second_c * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b
