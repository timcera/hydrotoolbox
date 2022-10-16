import numpy as np


def Eckhardt(Q, b_LH, a, BFImax, return_exceed=False):
    """Eckhardt filter (Eckhardt, 2005)

    Args:
        Q (np.array): streamflow
        a (float): recession coefficient
        BFImax (float): maximum value of baseflow index (BFI)
    """
    b = np.zeros(Q.shape[0] + 1) if return_exceed else np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    first_c = (1 - BFImax) * a
    second_c = (1 - a) * BFImax
    third_c = 1 - a * BFImax
    for i in range(Q.shape[0] - 1):
        b[i + 1] = (first_c * b[i] + second_c * Q[i + 1]) / (third_c)
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1
    return b


def f_Eckhardt(a):
    def _Eckhardt(Q, b_LH, BFImax, return_exceed=False):
        return Eckhardt(Q, b_LH, a, BFImax, return_exceed=return_exceed)

    return _Eckhardt
