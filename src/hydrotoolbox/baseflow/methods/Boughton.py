import numpy as np


def Boughton(Q, b_LH, a, C, return_exceed=False):
    """Boughton double-parameter filter (Boughton, 2004)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
        C (float): calibrated in baseflow.param_estimate
    """
    b = np.zeros(Q.shape[0] + 1) if return_exceed else np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    first_c = a / (1 + C)
    second_c = C / (1 + C)
    for i in range(Q.shape[0] - 1):
        b[i + 1] = first_c * b[i] + second_c * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b


def f_Boughton(a):
    def _Boughton(Q, b_LH, C, return_exceed=False):
        return Boughton(Q, b_LH, a, C, return_exceed=return_exceed)

    return _Boughton
