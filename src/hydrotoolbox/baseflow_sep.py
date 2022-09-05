# -*- coding: utf-8 -*-
"""Tools for hydrology.

hydrotoolbox baseflow_sep --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow_sep sliding < daily.csv
hydrotoolbox baseflow_sep eckardt,sliding < daily.csv
...

hydrotoolbox recession """

import logging
import warnings

import numpy as np
import pandas as pd
import typic
from toolbox_utils import tsutils

from .baseflow.comparison import strict_baseflow
from .baseflow.separation import separation

warnings.filterwarnings("ignore")


def bfsep(
    Q,
    method,
    print_input,
    bfi=False,
    date=None,
    area=None,
    ice_period=None,
    k=None,
    C=None,
    a=None,
):
    complete_index = pd.date_range(start=Q.index[0], end=Q.index[-1], freq="D")
    Q = Q.reindex(complete_index, fill_value=np.nan)
    if print_input is True:
        ntsd = Q.copy()
    Qb = pd.DataFrame()
    for col in Q.columns:
        ncol = Q[col].astype("float64")
        negmask = ncol <= 0
        if negmask.any():
            logging.warning(
                tsutils.error_wrapper(
                    f"""{negmask.sum()} negative or 0 values in input data.  No
                baseflow separation technique works with negative values.
                Negative values dropped from the analysis. This means that
                positive values on either side of negative flows are considered
                adjacent. Negative flow in the output baseflow represented
                as missing."""
                )
            )
        missingmask = ncol.isnull()
        if missingmask.any():
            logging.warning(
                tsutils.error_wrapper(
                    f"""{missingmask.sum()} missing values in input data.  No
                baseflow separation technique works with missing values.
                Missing values dropped from the analysis. This means that
                positive values on either side of missing flows are considered
                adjacent. Missing flow in the output baseflow represented
                as missing."""
                )
            )
        ncol[negmask] = pd.NA
        ncol = ncol.dropna()
        ndf = pd.DataFrame(
            separation(
                ncol.values,
                date=date,
                area=area,
                ice_period=ice_period,
                method=method,
                k=k,
                C=C,
                a=a,
            )[bfi],
            index=ncol.index,
        )
        ndf.columns = [col]
        Qb = Qb.join(ndf, how="outer")
    Qb = Qb.reindex(Q.index)
    return tsutils.return_input(print_input, ntsd, Qb, suffix=method.lower())


@tsutils.doc(tsutils.docstrings)
def boughton(
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
    """Boughton double-parameter filter (Boughton, 2004)

    ::

      |           C             k
      |  Qb   = -----  Q   +  -----  Qb
      |    i    1 + C   i     1 + C    (i-1)

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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "boughton", print_input)


@tsutils.doc(tsutils.docstrings)
def chapman(
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
    """Chapman filter (Chapman, 1991)

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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "chapman", print_input)


@tsutils.doc(tsutils.docstrings)
def cm(
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
    """CM filter (Chapman and Maxwell, 1996)

    ::

     |          1 - k           k
     |   Qb   = -----  Q   +  -----  Qb
     |     i    2 - k   i     2 - k    (i-1)

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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "cm", print_input)


@tsutils.doc(tsutils.docstrings)
def eckhardt(
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
    """Eckhardt filter (Eckhardt, 2005)

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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "eckhardt", print_input)


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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "ewma", print_input)


@tsutils.doc(tsutils.docstrings)
def usgs_hysep_fixed(
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
    area: float
        basin area in mile^2
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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "fixed", print_input, area=area)


@tsutils.doc(tsutils.docstrings)
def furey(
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
    """Furey digital filter (Furey and Gupta, 2001, 2003)

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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "furey", print_input)


@tsutils.doc(tsutils.docstrings)
def lh(
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
    """LH digital filter (Lyne and Hollick, 1979)

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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "lh", print_input)


@tsutils.doc(tsutils.docstrings)
def usgs_hysep_local(
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
    """USGS HYSEP Local minimum graphical method (Sloto and Crouse, 1996)

    Parameters
    ----------
    area: float
        basin area in mile^2
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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "local", print_input, area=area)


@typic.al
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

    Parameters
    ----------
    k: float
        k
        coefficient
    C: float
        C
        coefficient
    a: float
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
    """
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "ihacres", print_input, k=k, C=C, a=a)


@tsutils.doc(tsutils.docstrings)
def usgs_hysep_slide(
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
    area: float
        Area of watershed in miles**2
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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "slide", print_input, area=area)


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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "ukih", print_input)


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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "willems", print_input)


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
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "five_day", print_input)


@tsutils.doc(tsutils.docstrings)
def strict(
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
    """Return "strict" baseflow.

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
    ${tablefmt}
    """
    Q = tsutils.common_kwds(
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
    return bfsep(Q, "strict", print_input)
