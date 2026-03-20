from hydrotoolbox.baseflow.methods.general_form import general_form_digital_filter


def lyne_hollick(flow, k):
    """LH digital filter (Lyne & Hollick, 1979)"""
    alpha = k
    beta = (1 - k) / 2
    gamma = 1.0

    return general_form_digital_filter(flow, alpha, beta, gamma)
