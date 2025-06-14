import os

__all__ = [
    "boughton",
    "chapman",
    "cm",
    "eckhardt",
    "ewma",
    "five_day",
    "fixed",
    "furey",
    "ihacres",
    "lh",
    "local",
    "slide",
    "strict",
    "ukih",
    "willems",
    "recession_coefficient",
    "param_calibrate",
    "recession_period",
    "maximum_BFI",
    "separation",
    "strict_baseflow",
    "KGE",
]

from .comparison import KGE, strict_baseflow
from .methods import (
    boughton,
    chapman,
    cm,
    eckhardt,
    ewma,
    five_day,
    fixed,
    furey,
    ihacres,
    lh,
    local,
    slide,
    strict,
    ukih,
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
