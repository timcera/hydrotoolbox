import shlex
import subprocess
from io import StringIO
from unittest import TestCase

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from toolbox_utils import tsutils

from hydrotoolbox import hydrotoolbox


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


BFLIST = [
    dict(expected="chapman"),
    dict(expected="cm"),
    dict(expected="eckhardt"),
    dict(expected="ewma"),
    dict(expected="five_day"),
    dict(expected="furey"),
    dict(expected="ihacres"),
    dict(expected="lh"),
    dict(expected="strict"),
    dict(expected="ukih"),
    dict(expected="usgs_hysep_fixed"),
    dict(expected="usgs_hysep_local"),
    dict(expected="usgs_hysep_slide"),
    dict(expected="willems"),
]


class TestBaseflowSep:
    params = {"test_baseflow_sep": BFLIST, "test_baseflow_sep_cli_mean": BFLIST}

    def test_baseflow_sep(self, expected):
        args = f"hydrotoolbox baseflow_sep {expected} --input_ts=tests/data.csv"
        complete = subprocess.run(
            args, capture_output=True, text=True, check=True, shell=True
        )
        testf = pd.read_csv(StringIO(complete.stdout), parse_dates=True, index_col=0)
        base = pd.read_csv(
            f"tests/data_{expected}.csv",
            parse_dates=True,
            index_col=0,
            usecols=[0, 2],
        )
        assert_frame_equal(testf, base)

    def test_baseflow_sep_cli_mean(self, expected):
        testf = eval(
            f"hydrotoolbox.baseflow_sep.{expected}(input_ts='tests/data.csv').astype('Float64')"
        )
        base = pd.read_csv(
            f"tests/data_{expected}.csv", parse_dates=True, index_col=0, usecols=[0, 2]
        )
        assert_frame_equal(testf, base)
