import numpy as np


def Willems(Q, b_LH, a, w, return_exceed=False):
    """digital filter (Willems, 2009)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
        w (float): case-speciﬁc average proportion of the quick ﬂow
                   in the streamflow, calibrated in baseflow.param_estimate
    """
    b = np.zeros(Q.shape[0] + 1) if return_exceed else np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    v = (1 - w) * (1 - a) / (2 * w)
    first_c = (a - v) / (1 + v)
    second_c = v / (1 + v)
    for i in range(Q.shape[0] - 1):
        b[i + 1] = first_c * b[i] + second_c * (Q[i] + Q[i + 1])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b


def f_Willems(a):
    def _Willems(Q, b_LH, w, return_exceed=False):
        return Willems(Q, b_LH, a, w, return_exceed=return_exceed)

    return _Willems
