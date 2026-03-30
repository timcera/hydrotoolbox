__all__ = [
    "willems",
]

from hydrotoolbox.baseflow.methods.general_form import general_form_digital_filter


def willems(flow, b_LH, k, w):
    """Digital filter (Willems, 2009)"""
    v = (1 - w) * (1 - k) / (2 * w)
    alpha = (k - v) / (1 + v)
    beta = v / (1 + v)
    gamma = 1

    return general_form_digital_filter(flow, alpha, beta, gamma)


def f_Willems(a):
    def _Willems(Q, b_LH, w):
        return willems(Q, b_LH, a, w)

    return _Willems
