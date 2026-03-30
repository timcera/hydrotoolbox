"""Tools for hydrology.

hydrotoolbox baseflow_sep --area 59.1 linear < daily.csv
hydrotoolbox baseflow_sep sliding < daily.csv
hydrotoolbox baseflow_sep eckhardt,sliding < daily.csv
...

hydrotoolbox recession"""

__all__ = [
    "boughton",
    "chapman",
    "chapman_maxwell",
    "eckhardt",
    "ewma",
    "five_day",
    "furey",
    "ihacres",
    "lyne_hollick",
    "ukih",
    "usgs_hysep_fixed",
    "usgs_hysep_local",
    "usgs_hysep_slide",
    "willems",
]

import logging
import warnings

import numpy as np
import pandas as pd

__all__ = [
    "boughton",
    "chapman",
    "chapman_maxwell",
    "eckhardt",
    "ewma",
    "usgs_hysep_fixed",
    "furey",
    "lyne_hollick",
    "usgs_hysep_local",
    "ihacres",
    "usgs_hysep_slide",
    "ukih",
    "willems",
    "five_day",
]

try:
    from pydantic import validate_call
except ImportError:
    from pydantic import validate_arguments as validate_call

from .baseflow.separation import separation
from .toolbox_utils.src.toolbox_utils import tsutils

warnings.filterwarnings("ignore")

tsutils.docstrings["area"] = """area: float
        [optional, default is None, where N is then set to 5 days]

        Basin area in km^2.

        The area is used to estimate N days using the following equation:

        .. math::

            N = {0.38610216 * A}^{0.2}

        The equation in the HYSEP report expects the area in square miles, but
        the equation above used in hydrotoolbox is for square kilometers.
"""
tsutils.docstrings["num_days"] = """num_days: int
        [optional, default is None, where N is then set to 5 days]

        Override the calculation of N days using the area.  This is useful for
        testing the effect of different N days on the baseflow separation.
"""
tsutils.docstrings["k"] = """k
        [optional, default is None, where k will be calculated from the
        input data]

        Groundwater recession constant.  The value of k is between 0 and 1.
        The number is usually close to 1.
"""
tsutils.docstrings["bfi_max"] = """bfi_max
        [optional, default is None, where bfi_max will be calculated from the
        input data]
"""


def bfsep(
    flow,
    method,
    print_input,
    bfi=False,
    date=None,
    num_days=None,
    area=None,
    ice_period=None,
    k=None,
    c3c1=None,
    C=None,
    a=None,
    bfi_max=None,
):
    complete_index = pd.date_range(start=flow.index[0], end=flow.index[-1], freq="D")
    flow = flow.reindex(complete_index, fill_value=np.nan)
    ntsd = pd.DataFrame()
    if print_input is True:
        ntsd = flow.copy()
    q_base = pd.DataFrame()
    for col in flow.columns:
        ncol = flow[col].astype("float64")
        negmask = ncol <= 0
        if negmask.any():
            logging.warning(
                tsutils.error_wrapper(
                    f"""{negmask.sum()} negative or 0 values in input data.  No
                    baseflow separation technique works with negative values.
                    Negative values dropped from the analysis. This means that
                    positive values on either side of negative flows are
                    considered adjacent. Negative flow in the output baseflow
                    represented as missing.
                    """
                )
            )
        missingmask = ncol.isnull()
        if missingmask.any():
            logging.warning(
                tsutils.error_wrapper(
                    f"""{missingmask.sum()} missing values in input data.  No
                    baseflow separation technique works with missing values.
                    Missing values dropped from the analysis. This means that
                    positive values on either side of missing flows are
                    considered adjacent. Missing flow in the output baseflow
                    represented as missing.
                    """
                )
            )
        ncol[negmask] = pd.NA
        ncol = ncol.dropna()
        ndf = pd.DataFrame(
            separation(
                ncol.values,
                date=date,
                num_days=num_days,
                area=area,
                ice_period=ice_period,
                method=method,
                k=k,
                c3c1=c3c1,
                C=C,
                a=a,
                bfi_max=bfi_max,
            )[bfi],
            index=ncol.index,
        )
        ndf.columns = [col]
        q_base = q_base.join(ndf, how="outer")
    q_base = q_base.reindex(flow.index)
    q_base.index.name = "Datetime"
    return tsutils.return_input(print_input, ntsd, q_base, suffix=method.lower())


@tsutils.doc(tsutils.docstrings)
def boughton(
    input_ts="-",
    columns=None,
    k=None,
    C=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """
    Boughton double-parameter filter [1]_

    ::

        bₜ = [k⁄(1+C)] bₜ₋₁ + [C⁄(1+C)] Qₜ

        bₜ = baseflow for the current day
        bₜ₋₁ = baseflow for the previous day
        Qₜ = total stream flow for the current day
        k = groundwater recession constant
        C = watershed shape parameter

    Parameters
    ----------
    input_ts
        Streamflow
    ${columns}
    ${k}
    C
        [optional, default is None.]

        If None will be estimated from the flow data.
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}

    References
    ----------
    .. [1] Boughton W.C., 1993, A hydrograph-based model for
       estimating water yield of ungauged catchments, Institute of Engineers
       Australia National Conference. Publ. 93/14, pp. 317-324.
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "boughton", print_input, k=k, C=C)


@tsutils.doc(tsutils.docstrings)
def chapman(
    k=None,
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """
    Chapman filter [1]_
    ::

        bₜ = (3k-1)⁄(3-k) bₜ₋₁ + (1-k)⁄(3-k) (Qₜ + Qₜ₋₁)

        bₜ = baseflow for the current day
        bₜ₋₁ = baseflow for the previous day
        Qₜ = total stream flow for the current day
        Qₜ₋₁ = total stream flow for the previous day
        k = groundwater recession constant

    Parameters
    ----------
    ${k}
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}

    References
    ----------
    .. [1]  Chapman T. (1991) - Comment on evaluation of automated
       techniques for base flow and recession analyses, by RJ Nathan and TA
       McMahon. Water Resources Research, 27(7), pp. 1783-1784.
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "chapman", print_input, k=k)


@tsutils.doc(tsutils.docstrings)
def chapman_maxwell(
    k=None,
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """Digital filter (Chapman and Maxwell, 1996)
    ::

        bₜ = k⁄(2-k) bₜ₋₁ + (1-k)⁄(2-k) Qₜ

        bₜ = baseflow for the current day
        bₜ₋₁ = baseflow for the previous day
        Qₜ = total stream flow for the current day
        Qₜ₋₁ = total stream flow for the previous day
        k = groundwater recession constant

    Parameters
    ----------
    ${k}
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "chapman_maxwell", print_input, k=k)


@tsutils.doc(tsutils.docstrings)
def eckhardt(
    input_ts="-",
    columns=None,
    k=None,
    bfi_max=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """Eckhardt filter (Eckhardt, 2005)
    ::

        bₜ = [(1 - BFIₘₐₓ) k bₜ₋₁ + (1 - k) BFIₘₐₓ Qₜ]⁄[1 - (k BFIₘₐₓ)]

        bₜ = baseflow for the current day
        bₜ₋₁ = baseflow for the previous day
        Qₜ = total stream flow for the current day
        k = groundwater recession constant
        BFI_{max} = long-term ratio of baseflow to total streamflow
                 [values between 0 and 1]

    Parameters
    ----------
    input_ts
        Streamflow
    ${columns}
    ${k}
    ${bfi_max}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "eckhardt", print_input, k=k, bfi_max=bfi_max)


@tsutils.doc(tsutils.docstrings)
def ewma(
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """Exponential Weighted Moving Average (EWMA) filter (Tularam and Ilahee, 2008)

    Parameters
    ----------
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "ewma", print_input)


@tsutils.doc(tsutils.docstrings)
def usgs_hysep_fixed(
    num_days=None,
    area=None,
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """USGS HYSEP Fixed interval method.

    Sloto, Ronald A., and Michele Y. Crouse. “HYSEP: A Computer Program for
    Streamflow Hydrograph Separation and Analysis.” USGS Numbered Series.
    Water-Resources Investigations Report. Geological Survey (U.S.), 1996.
    http://pubs.er.usgs.gov/publication/wri964040

    Parameters
    ----------
    ${num_days}
    ${area}
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "fixed", print_input, area=area, num_days=num_days)


@tsutils.doc(tsutils.docstrings)
def furey(
    k=None,
    c3c1=None,
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """
    Furey digital filter [Furey and Gupta, 2001]

    This hydrograph separation filter, introduced in 2001, is based on a mass
    balance equation for baseflow through a hillside, and its construction is
    founded on a physical-statistical theory of low streamflows developed by
    Furey and Gupta.

    ::

        bₜ = k bₜ₋₁ + (1-k) c3c1 (Qₜ₋₁ - bₜ₋₁)

        bₜ = baseflow for the current day
        bₜ₋₁ = baseflow for the previous day
        Qₜ₋₁ = total stream flow for the previous day
        k = recession constant [values between 0 and 1]
        c3c1 = ratio of overland flow to groundwater flow, sometimes expressed
               as c3/c1 where c3 is the ratio of groundwater recharge to
               precipitation and c1 is the ratio of overland flow to
               precipitation.

    Parameters
    ----------
    ${k}
    c3c1
        [optional, default is None.]

        Value from 0.001 to 10.

        Ratio of overland flow to groundwater flow.  If set to None will be
        estimated from the flow data.
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}

    References
    ----------
    .. [Furey and Gupta, 2001] Furey, P. R., and V. K. Gupta. 2001.
       A physically based filter for separating base flow from streamflow time
       series, Water Resour. Res., 37(11), 2709–2722
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "furey", print_input, k=k, c3c1=c3c1)


@tsutils.doc(tsutils.docstrings)
def lyne_hollick(
    input_ts="-",
    alpha=0.925,
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """
    Digital filter [1]_
    ::

        bₜ = α bₜ₋₁ + (1-α)⁄2 (Qₜ + Qₜ₋₁)

        bₜ = baseflow for the current day
        bₜ₋₁ = baseflow for the previous day
        Qₜ = total stream flow for the current day
        Qₜ₋₁ = total stream flow for the previous day
        α = recession constant [values between 0 and 1]

    Parameters
    ----------
    input_ts
        Streamflow
    ${columns}
    alpha
        Catchment constant (value between 0 and 1).  Default is 0.925.
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}

    References
    ----------
    .. [1] Lyne V., Hollick M. (1979) - Stochastic time-variable
       rainfall-runoff modelling. Institute of Engineers Australia National
       Conference. Publ. 79/10, pp. 89-93.
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "lyne_hollick", print_input, k=alpha)


@tsutils.doc(tsutils.docstrings)
def usgs_hysep_local(
    num_days=None,
    area=None,
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """
    USGS HYSEP Local minimum graphical method (Sloto and Crouse, 1996)

    Parameters
    ----------
    ${num_days}
    ${area}
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "local", print_input, area=area, num_days=num_days)


@validate_call
@tsutils.doc(tsutils.docstrings)
def ihacres(
    k: float,
    C: float,
    a: float,
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """IHACRES

    Jakeman-Hornberger digital filter [1]_

    ::

        bₜ = k/(1 + C) bₜ₋₁ + C⁄(1 + C) (Qₜ + a Qₜ₋₁)

        bₜ = baseflow for the current day
        bₜ₋₁ = baseflow for the previous day
        Qₜ = total stream flow for the current day
        Qₜ₋₁ = total stream flow for the previous day

    Parameters
    ----------
    k : float
        k
        coefficient
    C : float
        C
        coefficient
    a : float
        a
        coefficient
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}

    References
    ----------
    .. [1] Jakeman, A. J., and G. M.Hornberger
       (1993), How much complexity is warranted in a rainfall-runoff model?,
       Water Resour. Res., 29(8), 2637–2649, doi:10.1029/93WR00877.
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "ihacres", print_input, k=k, C=C, a=a)


@tsutils.doc(tsutils.docstrings)
def usgs_hysep_slide(
    num_days=None,
    area=None,
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """USGS HYSEP sliding interval method

    The USGS HYSEP sliding interval method described in
    `Sloto and Crouse, 1996`

    The flow series is filter with scipy.ndimage.genericfilter1D using
    numpy.nanmin function over a window of size `size`

    Sloto, Ronald A., and Michele Y. Crouse. “HYSEP: A Computer Program for
    Streamflow Hydrograph Separation and Analysis.” USGS Numbered Series.
    Water-Resources Investigations Report. Geological Survey (U.S.), 1996.
    http://pubs.er.usgs.gov/publication/wri964040.

    Parameters
    ----------
    ${num_days}
    ${area}
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "slide", print_input, area=area, num_days=num_days)


@tsutils.doc(tsutils.docstrings)
def ukih(
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """Graphical method developed by UK Institute of Hydrology (UKIH, 1980)

    Parameters
    ----------
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "ukih", print_input)


@tsutils.doc(tsutils.docstrings)
def willems(
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """Digital filter (Willems, 2009)
    ::

        v = (1 - w) * (1 - k) / (2 * w)
        bₜ = (k - v)/(1 + v) bₜ₋₁ + v⁄(1 + v) (Qₜ + Qₜ₋₁)

        bₜ = baseflow for the current day
        bₜ₋₁ = baseflow for the previous day
        Qₜ = total stream flow for the current day
        Qₜ₋₁ = total stream flow for the previous day

    Parameters
    ----------
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "willems", print_input)


@tsutils.doc(tsutils.docstrings)
def five_day(
    input_ts="-",
    columns=None,
    source_units=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    target_units=None,
    print_input=False,
):
    """Value kept if less than 90 percent of adjacent 5-day blocks.

    Parameters
    ----------
    input_ts
        Streamflow
    ${columns}
    ${source_units}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    flow = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
    )
    return bfsep(flow, "five_day", print_input)


# @tsutils.doc(tsutils.docstrings)
# def strict(
#     input_ts="-",
#     columns=None,
#     source_units=None,
#     start_date=None,
#     end_date=None,
#     dropna="no",
#     clean=False,
#     round_index=None,
#     skiprows=None,
#     index_type="datetime",
#     names=None,
#     target_units=None,
#     print_input=False,
# ):
#     """Return "strict" baseflow.
#
#     Parameters
#     ----------
#     input_ts
#         Streamflow
#     ${columns}
#     ${source_units}
#     ${start_date}
#     ${end_date}
#     ${dropna}
#     ${clean}
#     ${round_index}
#     ${skiprows}
#     ${index_type}
#     ${names}
#     ${target_units}
#     ${tablefmt}
#     """
#     flow = tsutils.common_kwds(
#         tsutils.read_iso_ts(
#             input_ts,
#             skiprows=skiprows,
#             names=names,
#             index_type=index_type,
#         ),
#         start_date=start_date,
#         end_date=end_date,
#         pick=columns,
#         round_index=round_index,
#         dropna=dropna,
#         clean=clean,
#         source_units=source_units,
#         target_units=target_units,
#     )
#     return bfsep(flow, "strict", print_input)
