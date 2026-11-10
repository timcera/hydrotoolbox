"""Tests for Lyne-Hollick recurrence and pass handling."""

# Standard library imports
import subprocess
from io import StringIO

# Third party imports
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter, lfilter_zi

# First party imports
from hydrotoolbox import baseflow_sep
from hydrotoolbox.baseflow.methods.lyne_hollick import lyne_hollick


@pytest.mark.parametrize(
    ("passes", "expected"),
    [
        (1, [10.0, 1.0, 1.3375, 1.9871875, 2.5881484375]),
        (2, [1.3375, 1.0, 1.3375, 1.9871875, 2.5881484375]),
    ],
)
def test_clipped_values_feed_next_step(passes, expected):
    flow = np.array([10.0, 1.0, 10.0, 10.0, 10.0])
    original = flow.copy()

    actual = lyne_hollick(flow, k=0.925, passes=passes, b0_method="Q0")

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(flow, original)


@pytest.mark.parametrize("passes", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [0.0, 0.8, 0.925, 1.0])
def test_matches_baseflowx_for_positive_flows(passes, k):
    flow = np.array([5.0, 3.0, 4.0, 2.0, 6.0, 3.0, 4.0])
    expected = flow.copy()
    for pass_number in range(passes):
        if pass_number:
            expected = expected[::-1].copy()
        source = expected.copy()
        for i in range(1, len(source)):
            expected[i] = min(
                source[i],
                k * expected[i - 1] + (1 - k) / 2 * (source[i - 1] + source[i]),
            )
    if passes % 2 == 0:
        expected = expected[::-1]

    np.testing.assert_allclose(
        lyne_hollick(flow, k=k, passes=passes, b0_method="Q0"), expected
    )


def test_default_initialization_matches_scipy_without_clipping():
    flow = np.full(5, 10.0)
    k = 0.925
    beta = (1 - k) / 2
    b, a = [beta, beta], [1.0, -k]
    expected = lfilter(b, a, flow, zi=lfilter_zi(b, a))[0]

    np.testing.assert_allclose(lyne_hollick(flow, k, passes=1), expected)


@pytest.mark.parametrize("initial", ["Q0", "min", 2.0, "2.0", -1.0, "zi"])
def test_initial_value_is_bounded_before_feedback(initial):
    flow = np.array([1.0, 10.0])
    first = lyne_hollick(flow, 0.8, passes=1, b0_method=initial)
    initial_value = (
        0.9 + 0.1 * flow[0]
        if initial == "zi"
        else (float(initial) if initial not in ("Q0", "min") else 1.0)
    )
    clipped = min(max(initial_value, 0), flow[0])
    np.testing.assert_allclose(first, [clipped, 0.8 * clipped + 1.1])


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("initial", ["zi", "Q0", "min", "1.5"])
def test_public_api_and_cli_cap_each_step(tmp_path, passes, initial):
    flow = [10.0, 1.0, 10.0, 10.0, 10.0]
    path = tmp_path / "flow.csv"
    pd.DataFrame(
        {"flow": flow}, index=pd.date_range("2020-01-01", periods=len(flow))
    ).to_csv(path, index_label="Datetime")

    expected = lyne_hollick(np.array(flow), 0.925, passes, b0_method=initial)

    api = baseflow_sep.lyne_hollick(
        input_ts=str(path), alpha=0.925, passes=passes, b0_method=initial
    )
    api.index = pd.to_datetime(api.index)
    api.index.freq = "D"

    initial_args = [f"--b0_method={initial}"]
    result = subprocess.run(
        [
            "hydrotoolbox",
            "baseflow_sep",
            "lyne_hollick",
            f"--input_ts={path}",
            "--alpha=0.925",
            f"--passes={passes}",
            *initial_args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cli = pd.read_csv(StringIO(result.stdout), index_col=0)
    cli.index = pd.to_datetime(cli.index)
    cli.index.freq = "D"

    assert api.iloc[:, 0].to_numpy() == pytest.approx(expected)
    assert cli.iloc[:, 0].to_numpy() == pytest.approx(expected, rel=1e-5)
