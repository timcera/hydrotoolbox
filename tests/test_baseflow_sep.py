import shlex
import subprocess
from io import StringIO
from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal
from toolbox_utils import tsutils

from hydrotoolbox import hydrotoolbox


class TestBaseflowSep(TestCase):
    def test_baseflow_sep(self):
        for septype in (
            # "boughton",
            "chapman",
            "cm",
            "eckhardt",
            "ewma",
            "five_day",
            "usgs_hysep_fixed",
            "furey",
            "ihacres",
            "lh",
            "usgs_hysep_local",
            "usgs_hysep_slide",
            "strict",
            "ukih",
            "willems",
        ):
            args = f"hydrotoolbox baseflow_sep {septype} --input_ts=tests/data.csv"
            # args = shlex.split(args)
            complete = subprocess.run(
                args, capture_output=True, text=True, check=True, shell=True
            )
            testf = pd.read_csv(
                StringIO(complete.stdout), parse_dates=True, index_col=0
            )
            base = pd.read_csv(
                f"tests/data_{septype}.csv",
                parse_dates=True,
                index_col=0,
                usecols=[0, 2],
            )
            assert_frame_equal(testf, base)

    def test_baseflow_sep_cli_mean(self):
        for septype in (
            # "boughton",
            "chapman",
            "cm",
            "eckhardt",
            "ewma",
            "five_day",
            "fixed",
            "furey",
            "ihacres",
            "lh",
            "local",
            "slide",
            "strict",
            "ukih",
            "willems",
        ):
            testf = eval(
                f"hydrotoolbox.baseflow_sep.{septype}(input_ts='tests/data.csv').astype('Float64')"
            )
            base = pd.read_csv(
                f"tests/data_{septype}.csv", parse_dates=True, index_col=0
            )
            assert_frame_equal(testf, base)
