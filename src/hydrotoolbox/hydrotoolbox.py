# -*- coding: utf-8 -*-
"""Tools for hydrology.

hydrotoolbox baseflow --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow sliding < daily.csv
hydrotoolbox baseflow eckardt,sliding < daily.csv
...

hydrotoolbox recession """

from __future__ import absolute_import, division, print_function

import os.path
import sys
import warnings

import pandas as pd
from mando import Program
from mando.rst_text_formatter import RSTHelpFormatter
from scipy.ndimage import generic_filter, minimum_filter1d
from scipy.signal import lfilter
from scipy.stats import linregress
from tstoolbox import tsutils

from . import baseflow_sep
from .baseflow.comparison import strict_baseflow
from .baseflow.param_estimate import recession_coefficient

warnings.filterwarnings("ignore")

program = Program("hydrotoolbox", "0.0")

program.add_subprog("baseflow_sep")

program.add_subprog("baseflow_identify")


@program.baseflow_sep.command(
    "boughton", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.boughton)
def _boughton_cli(
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
    """Boughton double-parameter filter (Boughton, 2004)"""
    tsutils.printiso(
        baseflow_sep.boughton(
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


@program.baseflow_sep.command(
    "chapman", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.chapman)
def _chapman_cli(
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
        baseflow_sep.chapman(
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


@program.baseflow_sep.command("cm", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.copy_doc(baseflow_sep.cm)
def _cm_cli(
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
    """CM filter (Chapman and Maxwell, 1996)"""
    tsutils.printiso(
        baseflow_sep.cm(
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


@program.baseflow_sep.command(
    "eckhardt", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.eckhardt)
def _eckhardt_cli(
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
        baseflow_sep.eckhardt(
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


@program.baseflow_sep.command("ewma", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.copy_doc(baseflow_sep.ewma)
def _ewma_cli(
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
    tsutils.printiso(
        baseflow_sep.ewma(
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


@program.baseflow_sep.command(
    "five_day", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.five_day)
def _five_day_cli(
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
    tsutils.printiso(
        baseflow_sep.five_day(
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


@program.baseflow_sep.command(
    "furey", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.furey)
def _furey_cli(
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
    tsutils.printiso(
        baseflow_sep.furey(
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


@program.baseflow_sep.command("lh", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.copy_doc(baseflow_sep.lh)
def _lh_cli(
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
    tsutils.printiso(
        baseflow_sep.lh(
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


@program.baseflow_sep.command(
    "ihacres", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.ihacres)
def _ihacres_cli(
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
    tsutils.printiso(
        baseflow_sep.ihacres(
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


@program.baseflow_sep.command("ukih", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.copy_doc(baseflow_sep.ukih)
def _ukih_cli(
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
    tsutils.printiso(
        baseflow_sep.ukih(
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


@program.baseflow_sep.command(
    "willems", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.willems)
def _willems_cli(
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
    tsutils.printiso(
        baseflow_sep.willems(
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


@program.baseflow_sep.command(
    "usgs_hysep_fixed", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.usgs_hysep_fixed)
def _usgs_hysep_fixed_cli(
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
    tsutils.printiso(
        baseflow_sep.usgs_hysep_fixed(
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


@program.baseflow_sep.command(
    "usgs_hysep_local", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.usgs_hysep_local)
def _usgs_hysep_local_cli(
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
    tsutils.printiso(
        baseflow_sep.usgs_hysep_local(
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


@program.baseflow_sep.command(
    "usgs_hysep_slide", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.usgs_hysep_slide)
def _usgs_hysep_slide_cli(
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
    tsutils.printiso(
        baseflow_sep.usgs_hysep_slide(
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


@program.baseflow_sep.command(
    "strict", formatter_class=RSTHelpFormatter, doctype="numpy"
)
@tsutils.copy_doc(baseflow_sep.strict)
def _strict_cli(
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
    tsutils.printiso(
        baseflow_sep.strict(
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


@program.command("recession", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def _recession_cli(
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
        ),
        tablefmt=tablefmt,
    )


@program.command("indices", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def indices_cli(
    indice_codes,
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
    tablefmt="plain",
):
    """Calculate hydrologic indices.

    Note the 1.67-year flood threshold (Poff, 1996) that applies to indices
    FH11, DH22, DH23, DH24, TA3, and TH3 (below).  Compute the log10 of the
    peak annual flows. Compute the log10 of the daily flows for the peak annual
    flow days. Calculate the coefficients for a linear regression equation for
    logs of peak annual flow versus logs of average daily flow for peak days.
    Using the log peak flow for the 1.67 year recurrence interval (60th
    percentile) as input to the regression equation, predict the log10 of the
    average daily flow. The threshold is 10 to the log10 (average daily flow)
    power.

    +------+--------------------------------------------------------------------+
    | Code | Description                                                        |
    +======+====================================================================+
    | MA1  | Mean of the daily mean flow values for the entire flow record.     |
    |      | cubic feet per second—temporal                                     |
    +------+--------------------------------------------------------------------+
    | MA2  | Median of the daily mean flow values for the entire flow record.   |
    |      | cubic feet per second—temporal                                     |
    +------+--------------------------------------------------------------------+
    | MA3  | Mean (or median) of the coefficients of variation (standard        |
    |      | deviation/mean) for each year.  Compute the coefficient of         |
    |      | variation for each year of daily flows. Compute the mean of the    |
    |      | annual coefficients of variation.                                  |
    |      | percent—temporal                                                   |
    +------+--------------------------------------------------------------------+
    | MA4  | Standard deviation of the percentiles of the logs of the entire    |
    |      | flow record divided by the mean of percentiles of the logs.        |
    |      | Compute the log10 of the daily flows for the entire record.        |
    |      | Compute the 5th, 10th, 15th, 20th, 25th, 30th, 35th, 40th, 45th,   |
    |      | 50th, 55th, 60th, 65th, 70th, 75th, 80th, 85th, 90th, and 95th     |
    |      | percentiles for the logs of the entire flow record.                |
    |      | Percentiles are computed by interpolating between the ordered      |
    |      | (ascending) logs of the flow values. Compute the standard          |
    |      | deviation and mean for the percentile values. Divide the standard  |
    |      | deviation by the mean.                                             |
    |      | percent–spatial                                                    |
    +------+--------------------------------------------------------------------+
    | MA5  | The skewness of the entire flow record is computed as the mean for |
    |      | the entire flow record (MA1) divided by the median (MA2) for the   |
    |      | entire flow record.                                                |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA6  | Range in daily flows is the ratio of the 10-percent to 90-percent  |
    |      | exceedance values for the entire flow record. Compute the          |
    |      | 5-percent to 95-percent exceedance values for the entire           |
    |      | flow record. Exceedance is computed by interpolating between the   |
    |      | ordered (descending) flow values.  Divide the 10-percent           |
    |      | exceedance value by the 90-percent value.                          |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA7  | Range in daily flows is computed like MA6, except using the 20     |
    |      | percent and 80 percent exceedance values. Divide the 20 percent    |
    |      | exceedance value by the 80 percent value.                          |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA8  | Range in daily flows is computed like MA6, except using the        |
    |      | 25-percent and 75-percent exceedance values. Divide the 25-percent |
    |      | exceedance value by the 75-percent value.                          |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA9  | Spread in daily flows is the ratio of the difference between the   |
    |      | 90th and 10th percentile of the logs of the flow data to the log   |
    |      | of the median of the entire flow record. Compute the log10 of the  |
    |      | daily flows for the entire record.  Compute the 5th, 10th, 15th,   |
    |      | 20th, 25th, 30th, 35th, 40th, 45th, 50th, 55th, 60th, 65th, 70th,  |
    |      | 75th, 80th, 85th, 90th, and 95th percentiles for the logs of the   |
    |      | entire flow record. Percentiles are computed by interpolating      |
    |      | between the ordered (ascending) logs of the flow values.  Compute  |
    |      | MA9 as (90th –10th) /log10(MA2).                                   |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA10 | Spread in daily flows is computed like MA9, except using the 20th  |
    |      | and 80th percentiles.                                              |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA11 | Spread in daily flows is computed like MA9, except using the 25th  |
    |      | and 75th percentiles.                                              |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA12 | Means (or medians) of monthly flow values. Compute the means for   |
    | to   | each.  Means (or medians) of monthly flow values. Compute the      |
    | MA23 | means for each month over the entire flow record. For example,    |
    |      | MA12 is the mean of all January flow values over the entire record |
    |      | (cubic feet per second— temporal).                                 |
    +------+--------------------------------------------------------------------+
    | MA24 | Variability (coefficient of variation) of monthly flow values.     |
    | to   | Compute the standard deviation for each.  Variability (coefficient |
    | MA35 | of month in each year over the entire flow record. Divide the      |
    |      | standard deviation by the mean for each month. Average (or take    |
    |      | median of) these values for each month across all years.           |
    |      | percent—temporal                                                   |
    +------+--------------------------------------------------------------------+
    | MA36 | Variability across monthly flows. Compute the minimum, maximum,    |
    |      | and mean flows for each month in the entire flow record.  MA36 is  |
    |      | the maximum monthly flow minus the minimum monthly flow divided by |
    |      | the median monthly flow.                                           |
    |      | dimensionless-spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA37 | Variability across monthly flows. Compute the first (25th          |
    |      | percentile) and the third (75th percentile) quartiles (every month |
    |      | in dimensionless— the flow record). MA37 is the third quartile     |
    |      | minus the first quartile divided by the median of the monthly      |
    |      | means.                                                             |
    +------+--------------------------------------------------------------------+
    | MA38 | Variability across monthly flows. Compute the 10th and 90th        |
    |      | percentiles for the monthly means (every month in the flow         |
    |      | record). MA38 is the 90th percentile minus the 10th percentile     |
    |      | divided by the median of the monthly means.                        |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA39 | Variability across monthly flows. Compute the standard deviation   |
    |      | for the monthly means. MA39 is the standard deviation times 100    |
    |      | divided by the mean of the monthly means.                          |
    |      | percent—spatial                                                    |
    +------+--------------------------------------------------------------------+
    | MA40 | Skewness in the monthly flows. MA40 is the mean of the monthly     |
    |      | flow means minus the median of the monthly means divided by the    |
    |      | median of the monthly means.                                       |
    |      | dimensionles-sspatial                                              |
    +------+--------------------------------------------------------------------+
    | MA41 | Annual runoff. Compute the annual mean daily flows. MA41 is the    |
    |      | mean of the annual means divided by the drainage area.             |
    |      | cubic feet per second/ square mile—temporal                        |
    +------+--------------------------------------------------------------------+
    | MA42 | Variability across annual flows. MA42 is the maximum annual flow   |
    |      | minus the minimum annual flow divided by the median annual flow.   |
    |      | dimensionless-spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA43 | Variability across annual flows. Compute the first (25th           |
    |      | percentile) and third (75th percentile) quartiles and the 10th and |
    |      | 90th — percentiles for the annual means (every year in the flow    |
    |      | record). MA43 is the third quartile minus the first quartile       |
    |      | divided by the median of the annual means.                         |
    |      | dimensionless-spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA44 | Variability across annual flows. Compute the first (25th           |
    |      | percentile) and third (75th percentile) quartiles and the 10th and |
    |      | 90th percentiles for the annual means (every year in the flow      |
    |      | record). MA44 is the 90th percentile minus the 10th percentile     |
    |      | divided by the median of the annual means.                         |
    |      | dimensionless-spatial                                              |
    +------+--------------------------------------------------------------------+
    | MA45 | Skewness in the annual flows. MA45 is the mean of the annual flow  |
    |      | means minus the median of the annual means divided by the median   |
    |      | of the annual means.                                               |
    |      | dimensionless-spatial                                              |
    +------+--------------------------------------------------------------------+
    | ML1  | Mean (or median) of minimum flows for each month across all years. |
    | to   | Compute the minimums for each month over the entire flow record.   |
    | ML12 | For example, ML1 is the mean of the minimums of all | January flow |
    |      | values over the entire record.                                     |
    |      | cubic feet per second— temporal                                    |
    +------+--------------------------------------------------------------------+
    | ML13 | Variability (coefficient of variation) across minimum monthly flow |
    |      | values. Compute the mean and standard deviation for the minimum    |
    |      | monthly flows over the entire flow record. ML13 is the standard    |
    |      | deviation times 100 divided by the mean minimum monthly flow for   |
    |      | all years.                                                         |
    |      | percent—spatial                                                    |
    +------+--------------------------------------------------------------------+
    | ML14 | Compute the minimum annual flow for each year. ML14 is the mean of |
    |      | the ratios of minimum annual flows to the median flow for each     |
    |      | year.                                                              |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | ML15 | Low-flow index. ML15 is the mean of the ratios of minimum annual   |
    |      | flows to the mean flow for each year.                              |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | ML16 | Median of annual minimum flows. ML16 is the median of the ratios   |
    |      | of minimum annual flows to the median flow for each year.          |
    |      | dimensionless— temporal                                            |
    +------+--------------------------------------------------------------------+
    | ML17 | Base flow. Compute the mean annual flows. Compute the minimum of   |
    |      | a 7-day moving average flows for each year and divide them by the  |
    |      | mean annual flow for that year. ML17 is the mean (or median—Use    |
    |      | Preferenceset by using the Preference option) of those ratios.     |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | ML18 | Variability in base flow. Compute the standard deviation for the   |
    |      | ratios of 7-day moving average flows to mean annual flows for each |
    |      | year. ML18 is the standard deviation times 100 divided by the mean |
    |      | of the ratios.                                                     |
    |      | percent—spatial                                                    |
    +------+--------------------------------------------------------------------+
    | ML19 | Base flow. Compute the ratios of the minimum annual flow to mean   |
    |      | annual flow for each year. ML19 is the mean (or median) of these   |
    |      | ratios times 100.                                                  |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | ML20 | Base flow. Divide the daily flow record into 5-day blocks. Find    |
    |      | the minimum flow for each block. Assign the minimum flow as        |
    |      | a base flow for that block if 90 percent of that minimum flow is   |
    |      | less than the minimum flows for the blocks on either side.         |
    |      | Otherwise, set it to zero. Fill in the zero values using linear    |
    |      | interpolation. Compute the total flow for the entire record and    |
    |      | the total base flow for the entire record. ML20 is the ratio of    |
    |      | total flow to total base flow.                                     |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | ML21 | Variability across annual minimum flows. Compute the mean and      |
    |      | standard deviation for the annual minimum flows. ML21 is the       |
    |      | standard deviation times 100 divided by the mean.                  |
    |      | percent—spatial                                                    |
    +------+--------------------------------------------------------------------+
    | ML22 | Specific mean annual minimum flow. ML22 is the mean (or median) of |
    |      | the annual minimum flows divided by the drainage area.             |
    |      | cubic feet per second/square mile—temporal                         |
    +------+--------------------------------------------------------------------+
    | MH1  | Mean (or median) maximum flows for each month across all years.    |
    | to   | Compute the maximums for each month over the entire cubic feet per |
    | MH12 | flow record. For example, MH1 is the mean of the maximums of all   |
    |      | January flow values over the entire record.                        |
    |      | second—temporal                                                    |
    +------+--------------------------------------------------------------------+
    | MH13 | Variability (coefficient of variation) across maximum monthly flow |
    |      | values. Compute the mean and standard deviation for the maximum    |
    |      | monthly flows over the entire flow record. MH13 is the standard    |
    |      | deviation times 100 divided by the mean maximum monthly flow for   |
    |      | all years.                                                         |
    |      | percent—spatial                                                    |
    +------+--------------------------------------------------------------------+
    | MH14 | Median of annual maximum flows. Compute the annual maximum flows   |
    |      | from monthly maximum flows. Compute the ratio of annual maximum    |
    |      | flow to median annual flow for each year. MH14 is the median of    |
    |      | these ratios.                                                      |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | MH15 | High flow discharge index. Compute the 1-percent exceedance value  |
    |      | for the entire data record. MH15 is the 1-percent exceedance value |
    |      | divided by the median flow for the entire record.                  |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MH16 | High flow discharge index. Compute the 10-percent exceedance value |
    |      | for the entire data record. MH16 is the 10-percent exceedance      |
    |      | value divided by the median flow for the entire record.            |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MH17 | High flow discharge index. Compute the 25-percent exceedance value |
    |      | for the entire data record. MH17 is the 25-percent exceedance      |
    |      | value divided by the median flow for the entire record.            |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MH18 | Variability across annual maximum flows. Compute the logs (log10)  |
    |      | of the maximum annual flows. Find the standard percent—spatial     |
    |      | deviation and mean for these values. MH18 is the standard          |
    |      | deviation times 100 divided by the mean.                           |
    +------+--------------------------------------------------------------------+
    | MH19 | Skewness in annual maximum flows.                                  |
    |      | dimensionless—spatial                                              |
    +------+--------------------------------------------------------------------+
    | MH20 | Specific mean annual maximum flow. MH20 is the mean (or median) of |
    |      | the annual maximum flows divided by the drainage area.             |
    |      | cubic feet per second/square mile—temporal                         |
    +------+--------------------------------------------------------------------+
    | MH21 | High flow volume index. Compute the average volume for flow events |
    |      | above a threshold equal to the median flow for the entire record.  |
    |      | MH21 is the average volume divided by the median flow for the      |
    |      | entire record.                                                     |
    |      | days—temporal                                                      |
    +------+--------------------------------------------------------------------+
    | MH22 | High flow volume. Compute the average volume for flow events above |
    |      | a threshold equal to three times the median flow for the entire    |
    |      | record. MH22 is the average volume divided by the median flow for  |
    |      | the entire record.                                                 |
    |      | days—temporal                                                      |
    +------+--------------------------------------------------------------------+
    | MH23 | High flow volume. Compute the average volume for flow events above |
    |      | a threshold equal to seven times the median flow for the entire    |
    |      | record. MH23 is the average volume divided by the median flow for  |
    |      | the entire record.                                                 |
    |      | days—temporal                                                      |
    +------+--------------------------------------------------------------------+
    | MH24 | High peak flow. Compute the average peak flow value for flow       |
    |      | events above a threshold equal to the median flow for the entire   |
    |      | record. MH24 is the average peak flow divided by the median flow   |
    |      | for the entire record.                                             |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | MH25 | High peak flow.  Compute the average peak flow value for flow      |
    |      | events above a threshold equal to three times the median flow      |
    |      | for the entire record.  MH25 is the average peak flow divided by   |
    |      | the median flow for the entire record.                             |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | MH26 | High peak flow. Compute the average peak flow value for flow       |
    |      | events above a threshold equal to seven times the median flow for  |
    |      | the entire record. MH26 is the average peak flow divided by the    |
    |      | median flow for the entire record.                                 |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | MH27 | High peak flow.  Compute the average peak flow value for flow      |
    |      | events above a threshold equal to 75th-percentile value for the    |
    |      | entire flow record. MH27 is the average peak flow divided by the   |
    |      | median flow for the entire record.                                 |
    |      | dimensionless—temporal                                             |
    +------+--------------------------------------------------------------------+
    | FL1  | Low flood pulse count. Compute the average number of flow events   |
    |      | with flows below a threshold equal to the 25th-percentile value    |
    |      | for the entire flow record. FL1 is the average (or median) number  |
    |      | of events.                                                         |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FL2  | Variability in low pulse count. Compute the standard deviation in  |
    |      | the annual pulse counts for FL1. FL2 is 100 times the standard     |
    |      | deviation divided by the mean pulse count.                         |
    |      | percent—spatial                                                    |
    +------+--------------------------------------------------------------------+
    | FL3  | Frequency of low pulse spells. Compute the average number of flow  |
    |      | events with flows below a threshold equal to 5 percent of the mean |
    |      | flow value for the entire flow record. FL3 is the average (or      |
    |      | median) number of events.                                          |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FH1  | High flood pulse count. Compute the average number of flow events  |
    |      | with flows above a threshold equal to the 75th-percentile value    |
    |      | for the entire flow record. FH1 is the average (or median) number  |
    |      | of events.                                                         |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FH2  | Variability in high pulse count. Compute the standard deviation in |
    |      | the annual pulse counts for FH1. FH2 is 100 times the standard     |
    |      | deviation divided by the mean pulse count.                         |
    |      | number of events/year—spatial                                      |
    +------+--------------------------------------------------------------------+
    | FH3  | High flood pulse count. Compute the average number of days per     |
    |      | year that the flow is above a threshold equal to three times the   |
    |      | median flow for the entire record. FH3 is the mean (or median) of  |
    |      | the annual number of days for all years.                           |
    |      | number of days/year—temporal                                       |
    +------+--------------------------------------------------------------------+
    | FH4  | High flood pulse count. Compute the average number of days per     |
    |      | year that the flow is above a threshold equal to seven times the   |
    |      | median flow for the entire record. FH4 is the mean (or median) of  |
    |      | the annual number of days for all years.                           |
    |      | number of days/year—temporal                                       |
    +------+--------------------------------------------------------------------+
    | FH5  | Flood frequency. Compute the average number of flow events with    |
    |      | flows above a threshold equal to the median flow value for the     |
    |      | entire flow record. FH5 is the average (or median) number of       |
    |      | events.                                                            |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FH6  | Flood frequency. Compute the average number of flow events with    |
    |      | flows above a threshold equal to three times the median flow value |
    |      | for the entire flow record. FH6 is the average (or median) number  |
    |      | of events.                                                         |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FH7  | Flood frequency. Compute the average number of flow events with    |
    |      | flows above a threshold equal to seven times the median flow value |
    |      | for the entire flow record. FH6 is the average (or median) number  |
    |      | of events.                                                         |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FH8  | Flood frequency. Compute the average number of flow events with    |
    |      | flows above a threshold equal to 25-percent exceedance value for   |
    |      | the entire flow record. FH8 is the average (or median) number of   |
    |      | events.                                                            |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FH9  | Flood frequency. Compute the average number of flow events with    |
    |      | flows above a threshold equal to 75-percent exceedance value for   |
    |      | the entire flow record. FH9 is the average (or median) number of   |
    |      | events.                                                            |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FH10 | Flood frequency. Compute the average number of flow events with    |
    |      | flows above a threshold equal to median of the annual cubic        |
    |      | feet/second minima for the entire flow record. FH10 is the average |
    |      | (or median) number of events.                                      |
    |      | number of events/year—temporal                                     |
    +------+--------------------------------------------------------------------+
    | FH11 | Flood frequency. Compute the average number of flow events with    |
    |      | flows above a threshold equal to flow corresponding to a number of |
    |      | events/1.67-year recurrence interval (Poff, 1996; see index FH10   |
    |      | for computation details). FH11 is the average (or median) number   |
    |      | year—temporal                                                      |
    +------+--------------------------------------------------------------------+
    | DL1  | Annual minimum daily flow. Compute the minimum 1-day average flow  |
    |      | for each year. DL1 is the mean (or median) of these cubic feet per |
    |      | values. second—temporal                                            |
    +------+--------------------------------------------------------------------+
    | DL2  | Annual minimum of 3-day moving average flow. Compute the minimum   |
    |      | of a 3-day moving average flow for each year. DL2 cubic feet per   |
    |      | is the mean (or median) of these values. second—temporal           |
    +------+--------------------------------------------------------------------+
    | DL3  | Annual minimum of 7-day moving average flow. Compute the minimum   |
    |      | of a 7-day moving average flow for each year. DL3 cubic feet per   |
    |      | is the mean (or median) of these values. second—temporal           |
    +------+--------------------------------------------------------------------+
    | DL4  | Annual minimum of 30-day moving average flow. Compute the minimum  |
    |      | of a 30-day moving average flow for each year. cubic feet per      |
    |      | DL4 is the mean (or median) of these values. second—temporal       |
    +------+--------------------------------------------------------------------+

    Parameters
    ----------
    indice_codes
        A list of the hydrologic indice codes.
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
        [
            [key, val]
            for key, val in (
                indices(
                    indice_codes,
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
                )
            ).items()
        ],
        tablefmt=tablefmt,
        headers=["Indices", "Value"],
        float_format=".3f",
    )


@tsutils.transform_args(indice_codes=tsutils.make_list)
def indices(
    indice_codes,
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
    from .indices import indices as ind

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
    indice_class = ind.Indices(Q)

    hi = {}
    for icode in indice_codes:
        hi[icode] = getattr(indice_class, icode)()
    return hi


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
