# -*- coding: utf-8 -*-
import numpy as np


def Chapman(Q, b_LH, a, return_exceed=False):
    """Chapman filter (Chapman, 1991)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
    """
    b = [b_LH[0]]
    x = b_LH[0]
    for i in range(Q.shape[0] - 1):
        x = (3 * a - 1) / (3 - a) * x + (1 - a) / (3 - a) * (Q[i + 1] + Q[i])
        b.append(x)
    b = np.array(b)
    mask = b[1:] > Q[1:]
    b[1:][mask] = Q[1:][mask]
    if return_exceed:
        return np.append(b, np.count_nonzero(mask))
    return b
