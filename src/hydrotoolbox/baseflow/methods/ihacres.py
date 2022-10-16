"""Tools for hydrology.

hydrotoolbox baseflow_sep --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow_sep sliding < daily.csv
hydrotoolbox baseflow_sep eckardt,sliding < daily.csv
...

hydrotoolbox recession """


def ihacres(
    Q,
    k: float,
    C: float,
    a: float,
):
    Qb = Q.copy()
    first_c = k / (1 + C)
    second_c = C / (1 + C)
    for row in range(1, len(Q)):
        Qb[row] = first_c * Qb[row - 1] + second_c * (Q[row] + a * Q[row - 1])
    mask = Qb > Q
    Qb[mask] = Q[mask]
    return Qb
