__all__ = [
    "lyne_hollick",
]

from hydrotoolbox.baseflow.methods.general_form import general_form_digital_filter


def lyne_hollick(flow, k, passes=1):
    """LH digital filter (Lyne & Hollick, 1979)"""
    alpha = k
    beta = (1 - k) / 2
    gamma = 1.0

    for pss in range(passes):
        if pss > 0:
            flow = flow[::-1]
        flow, count = general_form_digital_filter(flow, alpha, beta, gamma)

    return flow, count
