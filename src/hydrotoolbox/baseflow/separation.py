import numpy as np

from .comparison import KGE, strict_baseflow
from .methods import (
    boughton,
    chapman,
    cm,
    eckhardt,
    ewma,
    f_Boughton,
    f_Eckhardt,
    f_Furey,
    f_Willems,
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
from .param_estimate import param_calibrate, recession_coefficient


def separation(
    Q,
    date=None,
    num_days=None,
    area=None,
    ice_period=None,
    method="all",
    k=None,
    c3c1=None,
    C=None,
    a=None,
    alpha=0.925,
    bfi_max=None,
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

    b = np.recarray(Q.shape[0], dtype=list(zip(method, [float] * len(method))))

    for m in method:
        if m in ["chapman", "cm", "boughton", "furey", "eckhardt", "willems"]:
            if k is None:
                k = recession_coefficient(Q, strict_bf, date, ice_period)
            k = float(k)

        if m in [
            "ukih",
            "local",
            "chapman",
            "cm",
            "boughton",
            "furey",
            "eckhardt",
            "willems",
            "ewma",
        ]:
            b_lh = lh(Q, alpha=alpha)[0]

        if m == "ukih":
            b[m] = ukih(Q, b_lh)

        if m == "local":
            b[m] = local(Q, b_lh, area=area, num_days=num_days)

        if m == "fixed":
            b[m] = fixed(Q, area=area, num_days=num_days)

        if m == "slide":
            b[m] = slide(Q, area=area, num_days=num_days)

        if m == "lh":
            b[m] = lh(Q, alpha=alpha)[0]

        if m == "chapman":
            b[m] = chapman(Q, b_lh, k)[0]

        if m == "cm":
            b[m] = cm(Q, b_lh, k)[0]

        if m == "boughton":
            if C is None:
                C = param_calibrate(
                    np.arange(0.0001, 1, 0.0001), f_Boughton(k), Q, b_lh
                )
            C = float(C)
            b[m] = boughton(Q, b_lh, k, C)[0]

        if m == "furey":
            if c3c1 is None:
                c3c1 = param_calibrate(np.arange(0.001, 10, 0.001), f_Furey(k), Q, b_lh)
            c3c1 = float(c3c1)
            b[m] = furey(Q, b_lh, k, c3c1)[0]

        if m == "eckhardt":
            if bfi_max is None:
                bfi_max = param_calibrate(
                    np.arange(0.0001, 1, 0.0001), f_Eckhardt(k), Q, b_lh
                )
            bfi_max = float(bfi_max)
            b[m] = eckhardt(Q, b_lh, k, bfi_max)[0]

        if m == "ewma":
            e = param_calibrate(np.arange(0.0001, 0.5, 0.0001), ewma, Q, b_lh)
            b[m] = ewma(Q, b_lh, e)[0]

        if m == "willems":
            w = param_calibrate(np.arange(0.0001, 1, 0.0001), f_Willems(k), Q, b_lh)
            b[m] = willems(Q, b_lh, k, w)[0]

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
