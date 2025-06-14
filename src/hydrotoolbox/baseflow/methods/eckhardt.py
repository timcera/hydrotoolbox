import numpy as np


def eckhardt(Q, b_LH, k, BFImax):
    """Eckhardt filter (Eckhardt, 2005)"""
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]

    first_c = (1 - BFImax) * k
    second_c = (1 - k) * BFImax
    third_c = 1 - k * BFImax

    num_exceed = 0
    for i in range(Q.shape[0] - 1):
        b[i + 1] = (first_c * b[i] + second_c * Q[i + 1]) / third_c
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            num_exceed += 1

    return b, num_exceed


def f_Eckhardt(a):
    def _Eckhardt(Q, b_LH, BFImax):
        return eckhardt(Q, b_LH, a, BFImax)

    return _Eckhardt
