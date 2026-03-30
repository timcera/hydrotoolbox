__all__ = [
    "chapman",
]

from hydrotoolbox.baseflow.methods.general_form import general_form_digital_filter


def chapman(flow, k):
    """Chapman filter (Chapman, 1991)"""
    alpha = (3 * k - 1) / (3 - k)
    beta = (1 - k) / (3 - k)
    gamma = 1.0

    return general_form_digital_filter(flow, alpha, beta, gamma)
