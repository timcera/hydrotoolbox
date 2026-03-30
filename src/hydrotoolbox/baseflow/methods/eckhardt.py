__all__ = [
    "eckhardt",
]

from hydrotoolbox.baseflow.methods.general_form import general_form_digital_filter


def eckhardt(flow, k, BFImax):
    """Eckhardt filter (Eckhardt, 2005)"""
    denom = 1 - k * BFImax
    alpha = ((1 - BFImax) * k) / denom
    beta = ((1 - k) * BFImax) / denom
    gamma = 0

    return general_form_digital_filter(flow, alpha, beta, gamma)


def f_Eckhardt(a):
    def _Eckhardt(Q, b_LH, BFImax):
        return eckhardt(Q, b_LH, a, BFImax)

    return _Eckhardt
