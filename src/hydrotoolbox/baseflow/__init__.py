# Standard library imports
import os

__all__ = [
    "KGE",
    "boughton",
    "chapman",
    "chapman_maxwell",
    "eckhardt",
    "ewma",
    "five_day",
    "furey",
    "ihacres",
    "lyne_hollick",
    "maximum_BFI",
    "param_calibrate",
    "recession_coefficient",
    "recession_period",
    "separation",
    "strict_baseflow",
    "ukih",
    "usgs_hysep_fixed",
    "usgs_hysep_local",
    "usgs_hysep_slide",
    "willems",
]

# Local folder imports
from .comparison import KGE, strict_baseflow
from .methods import (
    boughton,
    chapman,
    chapman_maxwell,
    eckhardt,
    ewma,
    five_day,
    furey,
    ihacres,
    lyne_hollick,
    ukih,
    usgs_hysep_fixed,
    usgs_hysep_local,
    usgs_hysep_slide,
    willems,
)
from .param_estimate import (
    maximum_BFI,
    param_calibrate,
    recession_coefficient,
    recession_period,
)
from .separation import separation

_path = os.path.dirname(__file__)
