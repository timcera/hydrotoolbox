import os

__all__ = [
    "Boughton",
    "Chapman",
    "CM",
    "Eckhardt",
    "EWMA",
    "five_day",
    "Fixed",
    "Furey",
    "ihacres",
    "LH",
    "Local",
    "Slide",
    "strict",
    "UKIH",
    "Willems",
    "recession_coefficient",
    "param_calibrate",
    "recession_period",
    "maximum_BFI",
    "Backward",
    "separation",
    "strict_baseflow",
    "KGE",
]

from .comparison import KGE, strict_baseflow
from .methods import (
    CM,
    EWMA,
    LH,
    UKIH,
    Boughton,
    Chapman,
    Eckhardt,
    Fixed,
    Furey,
    Local,
    Slide,
    Willems,
    five_day,
    ihacres,
    strict,
)
from .param_estimate import (
    Backward,
    maximum_BFI,
    param_calibrate,
    recession_coefficient,
    recession_period,
)
from .separation import separation

_path = os.path.dirname(__file__)
