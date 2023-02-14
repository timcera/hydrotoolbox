"""Tools for hydrology.

hydrotoolbox baseflow --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow sliding < daily.csv
hydrotoolbox baseflow eckardt,sliding < daily.csv
...

hydrotoolbox recession """

import datetime
import os.path
import re
import sys
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from cltoolbox import Program
from cltoolbox.rst_text_formatter import RSTHelpFormatter
from pydantic import validate_arguments
from scipy.signal import find_peaks
from toolbox_utils import tsutils

from . import baseflow_sep
from .baseflow.comparison import strict_baseflow
from .baseflow.param_estimate import recession_coefficient
from .indices import indices as ind

warnings.filterwarnings("ignore")

program = Program("hydrotoolbox", "0.0")

program.add_subprog("baseflow_sep")


def atoi(text):
    """Support for the natural_keys sorting function."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Sorting function for mixed alphanumeric labels in indices function."""
    return [atoi(c) for c in re.split(r"(\d+)", text)]


@program.baseflow_sep.command("boughton", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("chapman", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("cm", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("eckhardt", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("ewma", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("five_day", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("furey", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("lh", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("ihacres", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("ukih", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("willems", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("usgs_hysep_fixed", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("usgs_hysep_local", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("usgs_hysep_slide", formatter_class=RSTHelpFormatter)
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


@program.baseflow_sep.command("strict", formatter_class=RSTHelpFormatter)
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
    rc = {}
    for col in flow.columns:
        val = flow.loc[:, col].astype("float64").values
        strict = strict_baseflow(val)
        rc[col] = [recession_coefficient(val, strict, date, ice_period)]
    return rc


@program.command("recession", formatter_class=RSTHelpFormatter)
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


@program.command("flow_duration", formatter_class=RSTHelpFormatter)
@tsutils.doc(tsutils.docstrings)
def _flow_duration_cli(
    input_ts="-",
    exceedance_probabilities=(99.5, 99, 98, 95, 90, 75, 50, 25, 10, 5, 2, 1, 0.5),
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
    tablefmt="csv",
):
    """Flow duration.

    Parameters
    ----------
    ${input_ts}

    exceedance_probabilities
        [optional, default: (99.5, 99, 98, 95, 90, 75, 50, 25, 10, 5, 2, 1, 0.5)]

        Exceedance probabilities

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
        flow_duration(
            input_ts=input_ts,
            exceedance_probabilities=exceedance_probabilities,
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
        showindex=True,
    )


@tsutils.copy_doc(_flow_duration_cli)
def flow_duration(
    input_ts="-",
    exceedance_probabilities=(99.5, 99, 98, 95, 90, 75, 50, 25, 10, 5, 2, 1, 0.5),
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
    """Calculate flow duration for different exceedance probabilities."""
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
    exceedance_probabilities = np.array(exceedance_probabilities) / 100
    ndf = flow.quantile((1 - exceedance_probabilities), axis="rows")
    ndf.index = exceedance_probabilities
    ndf.index.name = "Quantiles"
    return ndf


@program.command("storm_events", formatter_class=RSTHelpFormatter)
@tsutils.doc(tsutils.docstrings)
def _storm_events_cli(
    rise_lag,
    fall_lag,
    input_ts="-",
    window=1,
    min_peak=0,
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
    """Storm events.

    Parameters
    ----------
    rise_lag : int
        Sets the number of time-series terms to include from the rising limb of
        the hydrograph.

    fall_lag : int
        Sets the number of time-series terms to include from the falling limb of
        the hydrograph. window=1

    min_peak : int, float
        [optional, default=0]

        All detected storm peaks in the hydrograph must be greater than
        `min_peak`.

    window : int
        [optional, default=1]

        Adjacent peaks can not be within `window` time-series terms of each
        other.

    ${input_ts}

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
        storm_events(
            rise_lag=rise_lag,
            fall_lag=fall_lag,
            input_ts=input_ts,
            window=window,
            min_peak=min_peak,
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


@tsutils.copy_doc(_storm_events_cli)
def storm_events(
    rise_lag,
    fall_lag,
    input_ts="-",
    min_peak=None,
    window=1,
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
    """Find peak storm events."""
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
    if min_peak is None:
        min_peak = flow.median()
    peaks, _ = find_peaks(
        flow.iloc[:, 0].astype("float64"), distance=window, height=min_peak
    )
    collected = set()
    for peak in peaks:
        collected.update(
            range(peak - int(float(rise_lag)), peak + int(float(fall_lag)) + 1)
        )
    index = sorted(list(collected))
    ndf = pd.DataFrame(flow.iloc[index, 0])
    ndf.columns = flow.columns
    return ndf


@program.command("indices", formatter_class=RSTHelpFormatter)
@tsutils.doc(tsutils.docstrings)
def _indices_cli(
    indice_codes,
    water_year="A-SEP",
    drainage_area=1,
    use_median=False,
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
        A list of the hydrologic indice codes, stream classifications, and/or
        flow regime indices to be computed.

        The hydrologic indice codes are taken as is, but the collected stream
        classifications are intersected with the flow regime indices.

    water_year
        [optional, default="A-SEP"]

        The water year to use for the calculation.  This uses the one of the
        "A-..." Pandas offset codes.  The "A-SEP" code represents the very end of
        September (the start of October) as the end of the water year.

    use_median : bool
        [optional, default=False]

        If True, use the median instead of the mean for the calculations.

    drainage_area
        [optional, default=1]

        The drainage area to use for the calculations.  This is the drainage
        area in square miles.

    ${input_ts}

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
                    water_year=water_year,
                    drainage_area=drainage_area,
                    use_median=use_median,
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
    *indice_codes,
    input_ts="-",
    water_year="A-SEP",
    drainage_area=1,
    use_median=False,
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
    """Return the requested hydrologic indices."""

    indice_codes = list(indice_codes)
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
    indice_class = ind.Indices(
        flow, water_year=water_year, use_median=use_median, drainage_area=drainage_area
    )

    sclasses = [
        "HARSH_INTERMITTENT",
        "FLASHY_INTERMITTENT",
        "SNOWMELT_PERENNIAL",
        "SNOW_RAIN_PERENNIAL",
        "GROUNDWATER_PERENNIAL",
        "FLASHY_PERENNIAL",
        "ALL_STREAMS",
    ]

    fcomps = [
        "AVERAGE_MAGNITUDE",
        "LOW_FLOW_MAGNITUDE",
        "HIGH_FLOW_MAGNITUDE",
        "LOW_FLOW_FREQUENCY",
        "HIGH_FLOW_FREQUENCY",
        "LOW_FLOW_DURATION",
        "HIGH_FLOW_DURATION",
        "TIMING",
        "RATE_OF_CHANGE",
    ]

    stream_classification = []
    flow_component = []
    for code in indice_codes:
        if code.upper() in sclasses:
            stream_classification.append(code)
        if code.upper() in fcomps:
            flow_component.append(code)

    for code in stream_classification + flow_component:
        indice_codes.remove(code)

    if stream_classification or flow_component:
        lu = {
            ("HARSH_INTERMITTENT", "AVERAGE_MAGNITUDE"): {"MA34", "MA22", "MA16"},
            ("HARSH_INTERMITTENT", "LOW_FLOW_MAGNITUDE"): {"ML13", "ML15", "ML1"},
            ("HARSH_INTERMITTENT", "HIGH_FLOW_MAGNITUDE"): {"MH23", "MH14", "MH9"},
            ("HARSH_INTERMITTENT", "LOW_FLOW_FREQUENCY"): {"FL2", "FL3", "FL1"},
            ("HARSH_INTERMITTENT", "HIGH_FLOW_FREQUENCY"): {"FH2", "FH5", "FH7"},
            ("HARSH_INTERMITTENT", "LOW_FLOW_DURATION"): {"DL1", "DL2", "DL3"},
            ("HARSH_INTERMITTENT", "HIGH_FLOW_DURATION"): {"DH5", "DH10"},
            ("HARSH_INTERMITTENT", "TIMING"): {"TH1", "TL2"},
            ("HARSH_INTERMITTENT", "RATE_OF_CHANGE"): {"RA4", "RA1", "RA5"},
            ("FLASHY_INTERMITTENT", "AVERAGE_MAGNITUDE"): {
                "MA37",
                "MA18",
                "MA21",
                "MA9",
            },
            ("FLASHY_INTERMITTENT", "LOW_FLOW_MAGNITUDE"): {
                "ML16",
                "ML6",
                "ML22",
                "ML15",
            },
            ("FLASHY_INTERMITTENT", "HIGH_FLOW_MAGNITUDE"): {
                "MH23",
                "MH4",
                "MH14",
                "MH7",
            },
            ("FLASHY_INTERMITTENT", "LOW_FLOW_FREQUENCY"): {"FL2", "FL3", "FL1"},
            ("FLASHY_INTERMITTENT", "HIGH_FLOW_FREQUENCY"): {
                "FH2",
                "FH3",
                "FH7",
                "FH10",
            },
            ("FLASHY_INTERMITTENT", "LOW_FLOW_DURATION"): {
                "DL1",
                "DL13",
                "DL16",
                "DL18",
            },
            ("FLASHY_INTERMITTENT", "HIGH_FLOW_DURATION"): {
                "DH12",
                "DH13",
                "DH15",
            },
            ("FLASHY_INTERMITTENT", "TIMING"): {"TA1", "TA2", "TL1"},
            ("FLASHY_INTERMITTENT", "RATE_OF_CHANGE"): {"RA9", "RA6", "RA5", "RA7"},
            ("SNOWMELT_PERENNIAL", "AVERAGE_MAGNITUDE"): {"MA29", "MA40"},
            ("SNOWMELT_PERENNIAL", "LOW_FLOW_MAGNITUDE"): {"ML13", "ML22"},
            ("SNOWMELT_PERENNIAL", "HIGH_FLOW_MAGNITUDE"): {"MH1", "MH20"},
            ("SNOWMELT_PERENNIAL", "LOW_FLOW_FREQUENCY"): {"FL2", "FL3"},
            ("SNOWMELT_PERENNIAL", "HIGH_FLOW_FREQUENCY"): {"FH8"},
            ("SNOWMELT_PERENNIAL", "LOW_FLOW_DURATION"): {"DL5", "DL16"},
            ("SNOWMELT_PERENNIAL", "HIGH_FLOW_DURATION"): {"DH16", "DH19"},
            ("SNOWMELT_PERENNIAL", "TIMING"): {"TA1"},
            ("SNOWMELT_PERENNIAL", "RATE_OF_CHANGE"): {"RA1", "RA8"},
            ("SNOW_RAIN_PERENNIAL", "AVERAGE_MAGNITUDE"): {"MA3", "MA44"},
            ("SNOW_RAIN_PERENNIAL", "LOW_FLOW_MAGNITUDE"): {"ML13", "ML14"},
            ("SNOW_RAIN_PERENNIAL", "HIGH_FLOW_MAGNITUDE"): {"MH17", "MH20"},
            ("SNOW_RAIN_PERENNIAL", "LOW_FLOW_FREQUENCY"): {"FL2", "FL3"},
            ("SNOW_RAIN_PERENNIAL", "HIGH_FLOW_FREQUENCY"): {"FH3", "FH5"},
            ("SNOW_RAIN_PERENNIAL", "LOW_FLOW_DURATION"): {"DL6", "DL13"},
            ("SNOW_RAIN_PERENNIAL", "HIGH_FLOW_DURATION"): {"DH12"},
            ("SNOW_RAIN_PERENNIAL", "TIMING"): {"TA1", "TL1"},
            ("SNOW_RAIN_PERENNIAL", "RATE_OF_CHANGE"): {"RA9", "RA8"},
            ("GROUNDWATER_PERENNIAL", "AVERAGE_MAGNITUDE"): {"MA3", "MA41", "MA8"},
            ("GROUNDWATER_PERENNIAL", "LOW_FLOW_MAGNITUDE"): {"ML18", "ML14", "ML16"},
            ("GROUNDWATER_PERENNIAL", "HIGH_FLOW_MAGNITUDE"): {"MH17", "MH19", "MH10"},
            ("GROUNDWATER_PERENNIAL", "LOW_FLOW_FREQUENCY"): {"FL2", "FL3", "FL1"},
            ("GROUNDWATER_PERENNIAL", "HIGH_FLOW_FREQUENCY"): {"FH3", "FH6"},
            ("GROUNDWATER_PERENNIAL", "LOW_FLOW_DURATION"): {"DL9", "DL11", "DL16"},
            ("GROUNDWATER_PERENNIAL", "HIGH_FLOW_DURATION"): {"DH11", "DH15", "DH20"},
            ("GROUNDWATER_PERENNIAL", "TIMING"): {"TA1", "TH1", "TL2"},
            ("GROUNDWATER_PERENNIAL", "RATE_OF_CHANGE"): {"RA9", "RA8", "RA5"},
            ("FLASHY_PERENNIAL", "AVERAGE_MAGNITUDE"): {"MA26", "MA41", "MA10"},
            ("FLASHY_PERENNIAL", "LOW_FLOW_MAGNITUDE"): {"ML17", "ML14", "ML16"},
            ("FLASHY_PERENNIAL", "HIGH_FLOW_MAGNITUDE"): {"MH23", "MH8", "MH14"},
            ("FLASHY_PERENNIAL", "LOW_FLOW_FREQUENCY"): {"FL2", "FL3"},
            ("FLASHY_PERENNIAL", "HIGH_FLOW_FREQUENCY"): {"FH4", "FH6", "FH7"},
            ("FLASHY_PERENNIAL", "LOW_FLOW_DURATION"): {"DL6", "DL10", "DL17"},
            ("FLASHY_PERENNIAL", "HIGH_FLOW_DURATION"): {"DH13", "DH16"},
            ("FLASHY_PERENNIAL", "TIMING"): {"TA1"},
            ("FLASHY_PERENNIAL", "RATE_OF_CHANGE"): {"RA9", "RA7", "RA6"},
            ("ALL_STREAMS", "AVERAGE_MAGNITUDE"): {"MA5", "MA41", "MA3", "MA11"},
            ("ALL_STREAMS", "LOW_FLOW_MAGNITUDE"): {"ML17", "ML4", "ML21", "ML18"},
            ("ALL_STREAMS", "HIGH_FLOW_MAGNITUDE"): {"MH16", "MH8", "MH10", "MH14"},
            ("ALL_STREAMS", "LOW_FLOW_FREQUENCY"): {"FL2", "FL3", "FL1"},
            ("ALL_STREAMS", "HIGH_FLOW_FREQUENCY"): {"FH2", "FH3", "FH6", "FH7"},
            ("ALL_STREAMS", "LOW_FLOW_DURATION"): {"DL13", "DL16", "DL17", "DL18"},
            ("ALL_STREAMS", "HIGH_FLOW_DURATION"): {"DH13", "DH15", "DH16", "DH20"},
            ("ALL_STREAMS", "TIMING"): {"TA1", "TL2"},
            ("ALL_STREAMS", "RATE_OF_CHANGE"): {"RA9", "RA8", "RA6", "RA5"},
        }

        for sclass in sclasses:
            lu[(sclass, None)] = set()

        for fcomp in fcomps:
            lu[(None, fcomp)] = set()

        for sclass in sclasses:
            for fcomp in fcomps:
                lu[(sclass, None)] = lu[(sclass, None)].union(lu[(sclass, fcomp)])
                lu[(None, fcomp)] = lu[(None, fcomp)].union(lu[(sclass, fcomp)])

        if not stream_classification:
            stream_classification = [None]
        if not flow_component:
            flow_component = [None]

        class_codes = []
        for stream_class in stream_classification:
            for flowc in flow_component:
                if stream_class is not None:
                    stream_class = stream_class.upper()
                if flowc is not None:
                    flowc = flowc.upper()
                class_codes.extend(lu[(stream_class, flowc)])
    else:
        class_codes = []

    description = {
        "MA1": "Mean of all daily flows",
        "MA2": "Median of all daily flows",
        "MA3": "CV of all daily flows",
        "MA4": "CV of  log of all daily flows",
        "MA5": "Mean daily flow/median daily flow",
        "MA6": "Q10/Q90 for all daily flows",
        "MA7": "Q20/Q80 for all daily flows",
        "MA8": "Q25/Q75 for all daily flows",
        "MA9": "(Q10-Q90)/median daily flow",
        "MA10": "(Q20-Q80)/median daily flow",
        "MA11": "(Q25-Q75)/median daily flow",
        "MA12": "Mean monthly flow: January",
        "MA13": "Mean monthly flow: February",
        "MA14": "Mean monthly flow: March",
        "MA15": "Mean monthly flow: April",
        "MA16": "Mean monthly flow: May",
        "MA17": "Mean monthly flow: June",
        "MA18": "Mean monthly flow: July",
        "MA19": "Mean monthly flow: August",
        "MA20": "Mean monthly flow: September",
        "MA21": "Mean monthly flow: October",
        "MA22": "Mean monthly flow: November",
        "MA23": "Mean monthly flow: December",
        "MA24": "CV of monthly flow: January",
        "MA25": "CV of monthly flow: February",
        "MA26": "CV of monthly flow: March",
        "MA27": "CV of monthly flow: April",
        "MA28": "CV of monthly flow: May",
        "MA29": "CV of monthly flow: June",
        "MA30": "CV of monthly flow: July",
        "MA31": "CV of monthly flow: August",
        "MA32": "CV of monthly flow: September",
        "MA33": "CV of monthly flow: October",
        "MA34": "CV of monthly flow: November",
        "MA35": "CV of monthly flow: December",
        "MA36": "Range mean monthly/median monthly flow",
        "MA37": "IQR mean monthly/median monthly flow",
        "MA38": "(Q10-Q90)[monthly]/median monthly flow",
        "MA39": "CV: monthly mean flows",
        "MA40": "Skewness in monthly flows",
        "MA41": "Mean annual runoff",
        "MA42": "Range mean annual/median annual flow",
        "MA43": "IQR mean annual/median annual flow",
        "MA44": "(Q10-Q90)[annual]/median annual flow",
        "MA45": "Skewness in annual flows",
        "ML1": "Mean minimum monthly flow: January",
        "ML2": "Mean minimum monthly flow: February",
        "ML3": "Mean minimum monthly flow: March",
        "ML4": "Mean minimum monthly flow: April",
        "ML5": "Mean minimum monthly flow: May",
        "ML6": "Mean minimum monthly flow: June",
        "ML7": "Mean minimum monthly flow: July",
        "ML8": "Mean minimum monthly flow: August",
        "ML9": "Mean minimum monthly flow: September",
        "ML10": "Mean minimum monthly flow: October",
        "ML11": "Mean minimum monthly flow: November",
        "ML12": "Mean minimum monthly flow: December",
        "ML13": "CV of minimum monthly flows",
        "ML14": "Mean minimum daily flow/mean median annual flow",
        "ML15": "Mean minimum annual flow/mean annual flow",
        "ML16": "Median minimum annual flow/median annual flow",
        "ML17": "7-day minimum flow/mean annual flow",
        "ML18": "CV of (7-day minimum flow/mean annual)",
        "ML19": "Mean of (minimum annual flow/mean annual)*100",
        "ML20": "Ratio of baseflow volume to total flow volume",
        "ML21": "CV of annual minimum flows",
        "ML22": "Mean annual minimum flow divided by catchment area",
        "MH1": "Mean maximum monthly flow: January",
        "MH2": "Mean maximum monthly flow: February",
        "MH3": "Mean maximum monthly flow: March",
        "MH4": "Mean maximum monthly flow: April",
        "MH5": "Mean maximum monthly flow: May",
        "MH6": "Mean maximum monthly flow: June",
        "MH7": "Mean maximum monthly flow: July",
        "MH8": "Mean maximum monthly flow: August",
        "MH9": "Mean maximum monthly flow: September",
        "MH10": "Mean maximum monthly flow: October",
        "MH11": "Mean maximum monthly flow: November",
        "MH12": "Mean maximum monthly flow: December",
        "MH13": "CV of maximum monthly flows",
        "MH14": "Median maximum annual flow/median annual flow",
        "MH15": "Mean of Q1 values/median daily flow across all years",
        "MH16": "Mean of Q10 values/median daily flow across all years",
        "MH17": "Mean of Q25 values/median daily flow across all years",
        "MH18": "CV of logarithmic annual maximum flows",
        "MH19": "Skewness in annual maximum flows",
        "MH20": "Mean annual maximum flow/catchment area",
        "MH21": "High-flow volume (thresh=1*median annual)",
        "MH22": "High-flow volume (thresh=3*median annual)",
        "MH23": "High-flow volume (thresh=7*median annual)",
        "MH24": "Maximum peak flow/median flow (thresh=1*median annual)",
        "MH25": "Maximum peak flow/median flow (thresh=3*median annual)",
        "MH26": "Maximum peak flow/median flow (thresh=7*median annual)",
        "MH27": "Maximum peak flow/median flow (threshold=Q25)",
        "FL1": "Annual low flow pulse count; number of periods<25th percentile",
        "FL2": "CV of low flow pulse count",
        "FL3": "Count of low flow spells (<5% of mean)/record length (yrs)",
        "FH1": "Annual high flow pulse count; number of periods>75th percentile",
        "FH2": "CV of high flow pulse count",
        "FH3": "Count of high flow events (>3*median annual)",
        "FH4": "Count of high flow events (>7*median annual)",
        "FH5": "Count of high flow events (>1*median annual)/record length (yrs)",
        "FH6": "Count of high flow events (>3*median annual)/record length (yrs)",
        "FH7": "Count of high flow events (>7*median annual)/record length (yrs)",
        "FH8": "Count of high flow events (>25th percentile)/record length (yrs)",
        "FH9": "Count of high flow events (>75th percentile)/record length (yrs)",
        "FH10": "Count of high flow events (>median of annual minima)/record length (yrs)",
        "FH11": "Mean number of discrete flood events per year",
        "DL1": "Annual minimum of 1-day mean of flow",
        "DL2": "Annual minimum of 3-day mean of flow",
        "DL3": "Annual minimum of 7-day mean of flow",
        "DL4": "Annual minimum of 30-day mean of flow",
        "DL5": "Annual minimum of 90-day mean of flow",
        "DL6": "CV: annual minimum of 1-day mean of flow",
        "DL7": "CV: annual minimum of 3-day mean of flow",
        "DL8": "CV: annual minimum of 7-day mean of flow",
        "DL9": "CV: annual minimum of 30-day mean of flow",
        "DL10": "CV: annual minimum of 90-day mean of flow",
        "DL11": "Mean of 1-day minimum of flow",
        "DL12": "Mean of 7-day minimum of flow",
        "DL13": "Mean of 30-day minimum of flow",
        "DL14": "Mean of Q75 values/median daily flow across all years",
        "DL15": "Mean of Q90 values/median daily flow across all years",
        "DL16": "Low flow pulse duration (Mean duration of FL1)",
        "DL17": "CV: low flow pulse duration (DL16)",
        "DL18": "Mean annual number of zero-flow days",
        "DL19": "CV: mean annual number of zero-flow days",
        "DL20": "Percentage of all months with zero flow",
        "DH1": "Annual maximum of 1-day mean of flow",
        "DH2": "Annual maximum of 3-day mean of flow",
        "DH3": "Annual maximum of 7-day mean of flow",
        "DH4": "Annual maximum of 30-day mean of flow",
        "DH5": "Annual maximum of 90-day mean of flow",
        "DH6": "CV: annual maximum of 1-day mean of flow",
        "DH7": "CV: annual maximum of 3-day mean of flow",
        "DH8": "CV: annual maximum of 7-day mean of flow",
        "DH9": "CV: annual maximum of 30-day mean of flow",
        "DH10": "CV: annual maximum of 90-day mean of flow",
        "DH11": "Mean of 1-day maximum of flow",
        "DH12": "Mean of 7-day maximum of flow",
        "DH13": "Mean of 30-day maximum of flow",
        "DH14": "Q95 value/mean monthly flow across all years",
        "DH15": "Mean duration of flood pulses>75th percentile flow",
        "DH16": "CV: mean duration of high flow pulse (FH1)",
        "DH17": "Mean duration of flood pulses>1*median flow",
        "DH18": "Mean duration of flood pulses>3*median flow",
        "DH19": "Mean duration of flood pulses>7*median flow",
        "DH20": "Mean duration of flood pulses>25th percentile of median flow",
        "DH21": "Mean duration of flood pulses>75th percentile of median flow",
        "DH22": "Mean annual median interval in days between floods over all years",
        "DH23": "Mean annual number of days that flows>threshold over all years",
        "DH24": "Mean annual maximum number of 365-day periods in which no floods occur",
        "TA1": "Constancy (see Colwell: 1974)",
        "TA2": "Predictability of flow",
        "TA3": "Seasonal predictability of flooding",
        "TL1": "Mean day-of-year of annual minimum",
        "TL2": "CV: day-of-year of annual minimum",
        "TL3": "Seasonal predictability of low flow",
        "TL4": "Seasonal predictability of non-low flow",
        "TH1": "Mean day-of-year of annual maximum",
        "TH2": "CV: day-of-year of annual maximum",
        "TH3": "Seasonal predictability of non-flooding",
        "RA1": "Mean of positive changes from one day to next (rise rate)",
        "RA2": "CV: mean of positive changes from one day to next (rise rate)",
        "RA3": "Mean of negative changes from one day to next (fall rate)",
        "RA4": "CV: mean of negative changes from one day to next (fall rate)",
        "RA5": "Ratio of days that are higher than previous day",
        "RA6": "Median of difference in log of flows over two consecutive days of rising flow",
        "RA7": "Median of difference in log of flows over two consecutive days of falling flow",
        "RA8": "Number of flow reversals from one day to the next",
        "RA9": "CV: number of flow reversals from one day to the next",
    }

    return {
        f"{icode}: {description[icode]}": getattr(indice_class, icode)()
        for icode in indice_codes + sorted(class_codes, key=natural_keys)
    }


@program.command("exceedance_time", formatter_class=RSTHelpFormatter)
@tsutils.doc(tsutils.docstrings)
def _exceedance_time_cli(
    input_ts="-",
    delays=0,
    under_over="over",
    time_units="day",
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
    *thresholds,
):
    """Calculate the time that a time series exceeds (or is below) a threshold.

    Calculate the time that a time series exceeds (or is below) a threshold.

    Parameters
    ----------
    *thresholds : list
        List of thresholds to calculate exceedance for.

    ${input_ts}

    delays : list
        [optional, default 0]

        List of delays to calculate exceedance for.  This can be an empty list
        in which case the delays are all 0.  If one delay is given, then each
        flow requires a delay term.

    under_over : str
        [optional, default "over"]

        Whether to calculate exceedance or under-exceedance.

    time_units : str
        [optional, default "day"]

        Units for the delays and the returned exceedance time.  Can be any
        of the following strings: "year", "month", "day", "hour", "min", or "sec".

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
    """
    ans = exceedance_time(
        input_ts=input_ts,
        delays=delays,
        under_over=under_over,
        time_units=time_units,
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
        *thresholds,
    )
    ans = list(ans.items())
    tsutils.printiso(
        ans,
        float_format=".3f",
        headers=["Flow", f"Exceedance Time ({under_over} {time_units})"],
    )


@validate_arguments
def exceedance_time(
    *thresholds,
    input_ts="-",
    delays=0,
    under_over="over",
    time_units: Literal["year", "month", "day", "hour", "min", "sec"] = "day",
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
    """Calculates the exceedance time over thresholds."""
    series = tsutils.common_kwds(
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

    year = datetime.timedelta(days=365, hours=6, minutes=9, seconds=9)
    punits = {
        "year": year,
        "month": year / 12,
        "day": datetime.timedelta(days=1),
        "hour": datetime.timedelta(hours=1),
        "min": datetime.timedelta(minutes=1),
        "sec": datetime.timedelta(seconds=1),
    }.get(time_units, time_units)

    series = pd.Series(series.iloc[:, 0])

    if isinstance(delays, (int, float)):
        delays = [delays]
    if delays == [0]:
        delays = [0] * len(thresholds)

    delays = [i * punits for i in delays]

    if len(delays) != len(thresholds):
        raise ValueError(
            tsutils.error_wrapper(
                """
                If any delay is given, then there must be a delay specified
                for each flow.
                """
            )
        )

    e_table = {}
    thresholds = [float(i) for i in thresholds]
    for flow, delay in zip(thresholds, delays):
        if under_over == "over":
            mask = series >= flow
        else:
            mask = series <= flow

        accum = datetime.timedelta(days=0)
        duration = datetime.timedelta(days=0)
        first = True
        for index, value in mask.iteritems():
            if pd.isna(value):
                continue
            if first is True:
                oindex = index
                ovalue = value
                first = False
                continue
            delta = index - oindex
            if ovalue is True and value is True:
                accum += delta
            elif ovalue is False and value is True:
                accum += (
                    (series[index] - flow) / (series[index] - series[oindex]) * delta
                )
            elif ovalue is True and value is False:
                accum += (
                    (series[oindex] - flow) / (series[oindex] - series[index]) * delta
                )
                duration = duration + max(datetime.timedelta(days=0), accum - delay)
                accum = datetime.timedelta(days=0)
            oindex = index
            ovalue = value
        duration = duration + max(datetime.timedelta(days=0), accum - delay)
        e_table[flow] = duration / punits
    return e_table


@program.command()
def about():
    """Display version number and system information."""
    tsutils.about(__name__)


def main():
    """Test for debug file."""
    if not os.path.exists("debug_hydrotoolbox"):
        sys.tracebacklimit = 0
    program()


if __name__ == "__main__":
    main()
