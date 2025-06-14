import subprocess
from io import StringIO

import pandas as pd
from pandas.testing import assert_frame_equal

from hydrotoolbox import hydrotoolbox

# This stupid thing is needed so that linters don't remove the hydrotoolbox
# import.
ht = hydrotoolbox


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


# Current test failures:
#     dict(expected="strict"),
#     dict(expected="ihacres"),
BFLIST = [
    {"expected": "ihacres"},
    {"expected": "chapman"},
    {"expected": "cm"},
    {"expected": "eckhardt"},
    {"expected": "ewma"},
    {"expected": "five_day"},
    {"expected": "furey"},
    {"expected": "lh"},
    {"expected": "ukih"},
    {"expected": "usgs_hysep_fixed"},
    {"expected": "usgs_hysep_local"},
    {"expected": "usgs_hysep_slide"},
    {"expected": "willems"},
]


class TestBaseflowSep:
    params = {"test_baseflow_sep": BFLIST, "test_baseflow_sep_cli_mean": BFLIST}

    def test_baseflow_sep(self, expected):
        input_fname = "tests/data_short.csv"
        extra_args = ""
        if expected == "ihacres":
            extra_args = "0.1 0.5 0.5"
        elif expected == "chapman":
            extra_args = "-k=0.5"
        elif expected == "cm":
            extra_args = "-k=0.1"
        elif expected == "lh":
            extra_args = "--alpha=0.750"
        elif expected in ["usgs_hysep_fixed", "usgs_hysep_local", "usgs_hysep_slide"]:
            extra_args = "--num_days=1"
        elif expected == "eckhardt":
            extra_args = "-k=0.83 --bfi_max=0.75"
        elif expected == "furey":
            extra_args = "-k=0.95 --c3c1=3.5"
        args = f"hydrotoolbox baseflow_sep {expected} --input_ts={input_fname} {extra_args}"
        complete = subprocess.run(
            args, capture_output=True, text=True, check=True, shell=True
        )
        testf = pd.read_csv(StringIO(complete.stdout), parse_dates=True, index_col=0)
        testf.columns = [f"Q::{expected}"]
        base = pd.read_csv(
            f"tests/data_baseflow_{expected}.csv",
            parse_dates=True,
            index_col=0,
        )
        base = base.iloc[:, -1].to_frame()
        base.index.name = "Datetime"
        base.columns = [f"Q::{expected}"]
        assert_frame_equal(testf, base, atol=1e-3)

    def test_baseflow_sep_cli_mean(self, expected):
        input_fname = "tests/data_short.csv"
        extra_args = ""
        if expected == "ihacres":
            extra_args = "0.1, 0.5, 0.5,"
        elif expected == "chapman":
            extra_args = "k=0.5,"
        elif expected == "cm":
            extra_args = "k=0.1,"
        elif expected == "lh":
            extra_args = "alpha=0.750,"
        elif expected in ["usgs_hysep_fixed", "usgs_hysep_local", "usgs_hysep_slide"]:
            extra_args = "num_days=1,"
        elif expected == "eckhardt":
            extra_args = "k=0.83, bfi_max=0.75,"
        elif expected == "furey":
            extra_args = "k=0.95, c3c1=3.5,"
        testf = eval(
            f"hydrotoolbox.baseflow_sep.{expected}({extra_args} input_ts='{input_fname}').astype('float64')"
        )
        testf.columns = [f"Q::{expected}"]
        base = pd.read_csv(
            f"tests/data_baseflow_{expected}.csv",
            parse_dates=True,
            index_col=0,
        ).asfreq("D")
        base = base.iloc[:, -1].to_frame()
        base.index.name = "Datetime"
        base.columns = [f"Q::{expected}"]
        assert_frame_equal(testf, base, check_index_type=False, atol=1e-3)
