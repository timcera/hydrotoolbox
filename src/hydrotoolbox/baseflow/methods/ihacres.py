"""Tools for hydrology.

hydrotoolbox baseflow_sep --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow_sep sliding < daily.csv
hydrotoolbox baseflow_sep eckhardt,sliding < daily.csv
...

hydrotoolbox recession"""


def ihacres(
    Q,
    k: float,
    C: float,
    a: float,
):
    """IHACRES baseflow separation.

    Jakeman-Hornberger digital filter (Jakeman and Hornberger, 1993)
    """
    Qb = Q.copy()

    alpha = k / (1 + C)
    beta = C / (1 + C)

    for row in range(1, len(Q)):
        Qb[row] = alpha * Qb[row - 1] + beta * (Q[row] + a * Q[row - 1])
    mask = Qb > Q
    Qb[mask] = Q[mask]
    return Qb
