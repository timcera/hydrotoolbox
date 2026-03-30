"""Tools for hydrology.

hydrotoolbox baseflow_sep --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow_sep sliding < daily.csv
hydrotoolbox baseflow_sep eckhardt,sliding < daily.csv
...

hydrotoolbox recession"""

__all__ = [
    "ihacres",
]

from hydrotoolbox.baseflow.methods.general_form import general_form_digital_filter


def ihacres(
    flow,
    k: float,
    C: float,
    a: float,
):
    """IHACRES baseflow separation.

    Jakeman-Hornberger digital filter (Jakeman and Hornberger, 1993)
    """
    alpha = k / (1 + C)
    beta = C / (1 + C)
    gamma = a

    return general_form_digital_filter(flow, alpha, beta, gamma)
