# -*- coding: utf-8 -*-
"""Tools for hydrology.

hydrotoolbox baseflow --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow sliding < daily.csv
hydrotoolbox baseflow eckardt,sliding < daily.csv
...

hydrotoolbox recession """

from __future__ import absolute_import, division, print_function

import importlib
import os.path
import sys
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from mando import Program
from mando.rst_text_formatter import RSTHelpFormatter
from scipy.ndimage import generic_filter, minimum_filter1d
from scipy.signal import lfilter
from scipy.stats import linregress
from tstoolbox import tsutils

from .baseflow.comparison import strict_baseflow
from .baseflow.param_estimate import recession_coefficient
from .baseflow.separation import separation
from .utils import nstar

warnings.filterwarnings("ignore")

program = Program("hydrotoolbox", "0.0")

program.add_subprog("baseflow_sep")


def bfsep(Q, method, print_input, bfi=False, date=None, area=None, ice_period=None):
    if print_input is True:
        ntsd = Q.copy()
    Qb = pd.DataFrame()
    for col in Q.columns:
        ndf = pd.DataFrame(
            separation(
                Q.loc[:, col].astype("float64").values,
                date=date,
                area=area,
                ice_period=ice_period,
                method=method,
            )[bfi],
            index=Q.index,
        )
        ndf.columns = [col]
        Qb = Qb.join(ndf, how="outer")
    return tsutils.return_input(print_input, ntsd, Qb, suffix=method.lower())


@program.baseflow_sep.command(
    "boughton", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.doc(tsutils.docstrings)
def boughton_cli(
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
    tablefmt="csv",
):
    """Boughton double-parameter filter (Boughton, 2004)

    !             C             k
    !    Qb   = -----  Q   +  -----  Qb
    !      i    1 + C   i     1 + C    (i-1)

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
    tsutils.printiso(
        boughton(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "Boughton", print_input)


@program.baseflow_sep.command(
    "chapman", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.doc(tsutils.docstrings)
def chapman_cli(
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
    tablefmt="csv",
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
    tsutils.printiso(
        chapman(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "Chapman", print_input)


@program.baseflow_sep.command("cm", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def cm_cli(
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
    tablefmt="csv",
):
    """CM filter (Chapman and Maxwell, 1996)

    !           1 - k           k
    !    Qb   = -----  Q   +  -----  Qb
    !      i    2 - k   i     2 - k    (i-1)

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
    tsutils.printiso(
        cm(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "CM", print_input)


@program.baseflow_sep.command(
    "eckhardt", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.doc(tsutils.docstrings)
def eckhardt_cli(
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
    tablefmt="csv",
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
    tsutils.printiso(
        eckhardt(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "Eckhardt", print_input)


@program.baseflow_sep.command("ewma", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def ewma_cli(
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
    tablefmt="csv",
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
    tsutils.printiso(
        ewma(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "EWMA", print_input)


@program.baseflow_sep.command(
    "fixed", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.doc(tsutils.docstrings)
def fixed_cli(
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
    tablefmt="csv",
):
    """Fixed interval method from USGS HYSEP program

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
    tsutils.printiso(
        fixed(
            area=area,
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


def fixed(
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
    return bfsep(Q, "Fixed", print_input, area=area)


@program.baseflow_sep.command(
    "furey", formatter_class=RSTHelpFormatter, doctype="numpy"
)
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
    tablefmt="csv",
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
    tsutils.printiso(
        furey(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "Furey", print_input)


@program.baseflow_sep.command("lh", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def lh_cli(
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
    tablefmt="csv",
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
    tsutils.printiso(
        lh(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "LH", print_input)


@program.baseflow_sep.command(
    "local", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.doc(tsutils.docstrings)
def local_cli(
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
    tablefmt="csv",
):
    """Local minimum graphical method from HYSEP program (Sloto and Crouse, 1996)

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
    tsutils.printiso(
        local(
            area=area,
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


def local(
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
    return bfsep(Q, "Local", print_input, area=area)


@program.baseflow_sep.command(
    "ihacres", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.doc(tsutils.docstrings)
def ihacres_cli(
    k,
    C,
    a,
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
    tablefmt="csv",
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
    tsutils.printiso(
        ihacres(
            k,
            C,
            a,
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


def ihacres(
    k,
    C,
    a,
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
    ntsd = pd.DataFrame()
    if print_input is True:
        ntsd = Q.copy()

    Qb = Q.copy()
    for col in range(len(Q.columns)):
        for row in range(1, len(Q.index)):
            Qb.iloc[row, col] = k / (1 + C) * Qb.iloc[row - 1, col] + C / (1 + C) * (
                Q.iloc[row, col] + a * Q.iloc[row - 1, col]
            )
    Qb.mask(Qb < Q, other=Q, inplace=True)
    return tsutils.return_input(print_input, ntsd, Qb, suffix="ihacres")


@program.baseflow_sep.command(
    "slide", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.doc(tsutils.docstrings)
def slide_cli(
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
    tablefmt="csv",
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
    tsutils.printiso(
        slide(
            area=area,
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


def slide(
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
    return bfsep(Q, "Slide", print_input, area=area)


@program.baseflow_sep.command("ukih", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def ukih_cli(
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
    tablefmt="csv",
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
    tsutils.printiso(
        ukih(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "UKIH", print_input)


@program.baseflow_sep.command(
    "willems", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.doc(tsutils.docstrings)
def willems_cli(
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
    tablefmt="csv",
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
    tsutils.printiso(
        willems(
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


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
    return bfsep(Q, "Willems", print_input)


@program.command("recession", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def recession_cli(
    date=None,
    ice_period=None,
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
    tablefmt="plain",
):
    """Recession coefficient.

    Parameters
    ----------
    date
        Date term
    ice_period
        Period of ice that changes the discharge relationship
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
    tsutils.printiso(
        recession(
            date=date,
            ice_period=ice_period,
            input_ts=input_ts,
            columns=columns,
            source_units=source_units,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


def recession(
    date=None,
    ice_period=None,
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
):
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
    rc = {}
    for col in Q.columns:
        val = Q.loc[:, col].astype("float64").values
        strict = strict_baseflow(val)
        rc[col] = [recession_coefficient(val, strict, date, ice_period)]
    return rc


@program.command()
def about():
    """Display version number and system information."""
    tsutils.about(__name__)


def main():
    """Set debug and run mando.main function."""
    if not os.path.exists("debug_hydrotoolbox"):
        sys.tracebacklimit = 0
    program()


if __name__ == "__main__":
    main()
