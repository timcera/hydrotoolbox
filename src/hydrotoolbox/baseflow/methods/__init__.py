__all__ = [
    "boughton",
    "chapman",
    "chapman_maxwell",
    "eckhardt",
    "ewma",
    "five_day",
    "fixed",
    "furey",
    "ihacres",
    "lyne_hollick",
    "local",
    "slide",
    "strict",
    "ukih",
    "willems",
    "f_Boughton",
    "f_Eckhardt",
    "f_Furey",
    "f_Willems",
]

from .boughton import boughton, f_Boughton
from .chapman import chapman
from .chapman_maxwell import chapman_maxwell
from .eckhardt import eckhardt, f_Eckhardt
from .ewma import ewma
from .five_day import five_day
from .fixed import fixed
from .furey import f_Furey, furey
from .ihacres import ihacres
from .local import local
from .lyne_hollick import lyne_hollick
from .slide import slide
from .strict import strict
from .ukih import ukih
from .willems import f_Willems, willems
