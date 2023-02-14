"""Tools for hydrology.

hydrotoolbox baseflow_sep --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow_sep sliding < daily.csv
hydrotoolbox baseflow_sep eckardt,sliding < daily.csv
...

hydrotoolbox recession """


from ..comparison import strict_baseflow


def strict(Q):
    Qb = strict_baseflow(Q)
    mask = Qb > Q
    Qb[mask] = Q[mask]
    return Qb
