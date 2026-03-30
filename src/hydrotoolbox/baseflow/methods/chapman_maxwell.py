__all__ = [
    "chapman_maxwell",
]

from hydrotoolbox.baseflow.methods.general_form import general_form_digital_filter


def chapman_maxwell(flow, k):
    """CM filter (Chapman & Maxwell, 1996)"""
    alpha = k / (2 - k)
    beta = (1 - k) / (2 - k)
    gamma = 0

    return general_form_digital_filter(flow, alpha, beta, gamma)
