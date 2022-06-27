# -*- coding: utf-8 -*-
import numpy as np


def Furey(Q, b_LH, a, A, return_exceed=False):
    """Furey digital filter (Furey & Gupta, 2001, 2003)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
        A (float): calibrated in baseflow.param_estimate
    """
    b = np.zeros(Q.shape[0] + 1) if return_exceed else np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = (a - A * (1 - a)) * b[i] + A * (1 - a) * Q[i]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b


def f_Furey(a):
    def _Furey(Q, b_LH, A, return_exceed=False):
        return Furey(Q, b_LH, a, A, return_exceed=return_exceed)

    return _Furey
