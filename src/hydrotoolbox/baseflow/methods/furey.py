import numpy as np


def furey(flow, b_LH, k, c3c1):
    """Furey digital filter (Furey & Gupta, 2001, 2003)"""
    b = np.zeros(flow.shape[0])

    b[0] = b_LH[0]

    num_exceed = 0
    for i in range(1, flow.shape[0]):
        b[i] = k * b[i - 1] + (1 - k) * c3c1 * (flow[i - 1] - b[i - 1])
        if b[i] > flow[i]:
            b[i] = flow[i]
            num_exceed += 1

    return b, num_exceed


def f_Furey(a):
    def _Furey(Q, b_LH, A):
        return furey(Q, b_LH, a, A)

    return _Furey
