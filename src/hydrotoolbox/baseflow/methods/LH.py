import numpy as np


def LH(Q, beta=0.925, return_exceed=False):
    """LH digital filter (Lyne & Hollick, 1979)

    Args:
        Q (np.array): streamflow
        beta (float): filter parameter, 0.925 recommended by (Nathan & McMahon, 1990)
    """
    b = np.zeros(Q.shape[0] + 1) if return_exceed else np.zeros(Q.shape[0])

    first_c = (1 - beta) / 2

    # first pass
    b[0] = Q[0] / 2
    for i in range(Q.shape[0] - 1):
        b[i + 1] = beta * b[i] + first_c * (Q[i] + Q[i + 1])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1

    # second pass
    b1 = np.copy(b)
    for i in range(Q.shape[0] - 2, -1, -1):
        b[i] = beta * b[i + 1] + first_c * (b1[i + 1] + b1[i])
        if b[i] > Q[i]:
            b[i] = Q[i]
            if return_exceed:
                b[-1] += 1
    return b
