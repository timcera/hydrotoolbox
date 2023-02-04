import numpy as np

from .comparison import KGE, strict_baseflow
from .methods import *
from .param_estimate import param_calibrate, recession_coefficient


def separation(
    Q, date=None, area=None, ice_period=None, method="all", k=None, C=None, a=None
):
    if method == "all":
        method = [
            "ukih",
            "local",
            "fixed",
            "slide",
            "lh",
            "chapman",
            "cm",
            "boughton",
            "furey",
            "eckhardt",
            "ewma",
            "willems",
            "ihacres",
            "strict",
            "five_day",
        ]
    elif isinstance(method, str):
        method = [method]

    strict_bf = strict_baseflow(Q)
    if any(
        m in ["chapman", "cm", "boughton", "furey", "eckhardt", "willems"]
        for m in method
    ):
        a = recession_coefficient(Q, strict_bf, date, ice_period)

    b_lh = LH(Q)
    b = np.recarray(Q.shape[0], dtype=list(zip(method, [float] * len(method))))
    for m in method:
        if m == "ukih":
            b[m] = UKIH(Q, b_lh)

        if m == "local":
            b[m] = Local(Q, b_lh, area)

        if m == "fixed":
            b[m] = Fixed(Q, area)

        if m == "slide":
            b[m] = Slide(Q, area)

        if m == "lh":
            b[m] = b_lh

        if m == "chapman":
            b[m] = Chapman(Q, b_lh, a)

        if m == "cm":
            b[m] = CM(Q, b_lh, a)

        if m == "boughton":
            C = param_calibrate(np.arange(0.0001, 1, 0.0001), f_Boughton(a), Q, b_lh)
            b[m] = Boughton(Q, b_lh, a, C)

        if m == "furey":
            A = param_calibrate(np.arange(0.001, 10, 0.001), f_Furey(a), Q, b_lh)
            b[m] = Furey(Q, b_lh, a, A)

        if m == "eckhardt":
            bfi_max = param_calibrate(
                np.arange(0.0001, 1, 0.0001), f_Eckhardt(a), Q, b_lh
            )
            b[m] = Eckhardt(Q, b_lh, a, bfi_max)

        if m == "ewma":
            e = param_calibrate(np.arange(0.0001, 0.5, 0.0001), EWMA, Q, b_lh)
            b[m] = EWMA(Q, b_lh, e)

        if m == "willems":
            w = param_calibrate(np.arange(0.0001, 1, 0.0001), f_Willems(a), Q, b_lh)
            b[m] = Willems(Q, b_lh, a, w)

        if m == "ihacres":
            b[m] = ihacres(Q, k=k, C=C, a=a)

        if m == "strict":
            b[m] = strict(Q)

        if m == "five_day":
            b[m] = five_day(Q)

    KGEs = KGE(
        b[strict_bf].view(np.float64).reshape(-1, len(method)),
        np.repeat(Q[strict_bf], len(method)).reshape(-1, len(method)),
    )
    return b, KGEs
