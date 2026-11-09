# Standard library imports
import inspect

# Third party imports
import numpy as np

# Local folder imports
from . import methods
from .comparison import KGE, strict_baseflow
from .param_estimate import recession_coefficient

ALL_METHODS = [
    "ukih",
    "usgs_hysep_local",
    "usgs_hysep_fixed",
    "usgs_hysep_slide",
    "lyne_hollick",
    "chapman",
    "chapman_maxwell",
    "boughton",
    "furey",
    "eckhardt",
    "ewma",
    "willems",
    "ihacres",
    "five_day",
]


def separation(Q, method="all", **params):
    """Run one or more baseflow methods on Q and score them with KGE.

    Parameters that are None or that a method does not accept are ignored,
    except that an explicit None for b0_method is rejected by the method.
    One set of parameters can be shared by every method.
    """
    if method == "all":
        method = ALL_METHODS
    elif isinstance(method, str):
        method = [method]

    strict_bf = strict_baseflow(Q)
    if params.get("k") is None:
        params["k"] = float(recession_coefficient(Q, strict_bf))

    b = np.recarray(Q.shape[0], dtype=list(zip(method, [float] * len(method))))
    for m in method:
        func = getattr(methods, m)
        accepted = inspect.signature(func).parameters
        kwargs = {
            key: val
            for key, val in params.items()
            if key in accepted and (val is not None or key == "b0_method")
        }
        b[m] = func(Q, **kwargs)

    KGEs = KGE(
        b[strict_bf].view(np.float64).reshape(-1, len(method)),
        np.repeat(Q[strict_bf], len(method)).reshape(-1, len(method)),
    )
    return b, KGEs
