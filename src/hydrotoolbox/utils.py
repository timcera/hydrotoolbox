"""Utilities for other functions in hydrotoolbox."""

import math

from pint import UnitRegistry

ureg = UnitRegistry()


def nstar(area=None, area_unit="mile**2"):
    """Estimate the duration of surface runoff.

    Is calculated from the empirical relation: N=A^0.2, (1) where N is the
    number of days after which surface runoff ceases, and A is the drainage
    area in square miles (Linsley and others, 1982, p. 210).  The interval 2N*
    used for hydrograph separations is the odd integer between 3 and 11 nearest
    to 2N (Pettyjohn and Henning, 1979, p. 31)."""
    if area is None:
        numdays = 5
    else:
        area = area * ureg(area_unit)
        area = area.to("miles**2").magnitude
        numdays = area**0.2
    inn = math.ceil(2 * numdays)
    if (inn % 2) == 0:
        inn = inn - 1
    inn = int(min(max(inn, 3), 11))
    return inn
