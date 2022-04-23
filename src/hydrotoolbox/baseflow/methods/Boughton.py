# -*- coding: utf-8 -*-
import numpy as np


def Boughton(Q, b_LH, a, C, return_exceed=False):
    """Boughton doulbe-parameter filter (Boughton, 2004)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
        C (float): calibrated in baseflow.param_estimate
    """
    b = [b_LH[0]]
    x = b_LH[0]
    for i in range(Q.shape[0] - 1):
        x = a / (1 + C) * x + C / (1 + C) * Q[i + 1]
        b.append(x)
    b = np.array(b)
    mask = b > Q
    b[mask] = Q[mask]
    if return_exceed:
        b = np.append(b, np.count_nonzero(mask))
    return b


def f_Boughton(a):
    def _Boughton(Q, b_LH, C, return_exceed=False):
        return Boughton(Q, b_LH, a, C, return_exceed=return_exceed)

    return _Boughton
