"""Tests for initial baseflow selection and propagation."""

# Standard library imports
import importlib
import subprocess
from io import StringIO

# Third party imports
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from scipy.signal import lfilter, lfilter_zi

# First party imports
from hydrotoolbox import baseflow_sep
from hydrotoolbox.baseflow.comparison import strict_baseflow
from hydrotoolbox.baseflow.general_form import general_form_digital_filter
from hydrotoolbox.baseflow.methods import (
    boughton,
    chapman,
    chapman_maxwell,
    eckhardt,
    ewma,
    furey,
    ihacres,
    lyne_hollick,
    willems,
)
from hydrotoolbox.baseflow.methods.lyne_hollick import lyne_hollick as lh_filter
from hydrotoolbox.baseflow.methods.ukih import ukih, ukih_turn
from hydrotoolbox.baseflow.methods.usgs_hysep_local import (
    hysep_interval,
    local_turn,
    usgs_hysep_local,
)
from hydrotoolbox.baseflow.param_estimate import (
    param_calibrate,
    recession_coefficient,
    resolve_k,
)
from hydrotoolbox.baseflow.separation import separation
from hydrotoolbox.baseflow.utils import _init_baseflow

FLOW = np.array([5.0, 3.0, 4.0, 2.0, 6.0, 3.0, 4.0])
K = 0.8
METHODS = [
    (boughton, {"C": 0.4}),
    (chapman, {}),
    (chapman_maxwell, {}),
    (eckhardt, {"bfi_max": 0.5}),
    (ewma, {"e": 0.2}),
    (furey, {"c3c1": 0.4}),
    (ihacres, {"C": 0.4, "a": 0.5}),
    (lyne_hollick, {}),
    (willems, {"w": 0.4}),
]


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
@pytest.mark.parametrize("initial", ["Q0", "min", "LH", 1.5, "1.5"])
def test_method_initial_value(method, kwargs, initial):
    expected = _init_baseflow(FLOW, initial, k=K)
    if method is lyne_hollick:
        kwargs = {**kwargs, "passes": 1}
    result = method(FLOW, k=K, b0_method=initial, **kwargs)
    assert result[0] == pytest.approx(expected)


def test_lh_uses_requested_recession_coefficient():
    flow = FLOW.copy()
    flow[0] = 10
    expected = lyne_hollick(flow, K)[0]
    assert _init_baseflow(flow, "LH", k=K) == pytest.approx(expected)
    assert expected != pytest.approx(lyne_hollick(flow, 0.5)[0])


def test_resolve_k_and_lh_with_estimated_recession():
    flow = pd.read_csv("tests/data_short.csv", index_col=0).iloc[:200, 0].to_numpy()
    expected_k = recession_coefficient(flow, strict_baseflow(flow))
    assert resolve_k(flow, None) == pytest.approx(expected_k)
    assert resolve_k(flow, "0.8") == pytest.approx(0.8)
    assert _init_baseflow(flow, "LH") == pytest.approx(
        lyne_hollick(flow, expected_k)[0]
    )


def test_invalid_initial_value():
    with pytest.raises(ValueError, match="Invalid initial_method"):
        furey(FLOW, k=K, c3c1=0.4, b0_method="unknown")


def test_general_filter_initial_condition_and_legacy_default():
    alpha, beta, gamma = 0.6, 0.2, 0.5
    numerator = [beta, beta * gamma]
    denominator = [1.0, -alpha]
    old = lfilter(numerator, denominator, FLOW, zi=lfilter_zi(numerator, denominator))[
        0
    ]
    result, count = general_form_digital_filter(FLOW, alpha, beta, gamma)
    assert result == pytest.approx(np.clip(old, 0, FLOW))
    assert count == np.count_nonzero(old > FLOW)

    raw = np.empty_like(FLOW)
    raw[0] = 1.5
    for i in range(1, len(FLOW)):
        raw[i] = alpha * raw[i - 1] + beta * (FLOW[i] + gamma * FLOW[i - 1])
    result, count = general_form_digital_filter(FLOW, alpha, beta, gamma, b0_method=1.5)
    assert result == pytest.approx(np.clip(raw, 0, FLOW))
    assert count == np.count_nonzero(raw > FLOW)


@pytest.mark.parametrize(
    ("alpha", "beta", "gamma", "delta"),
    [(0.8, 0.1, 1.0, 1.0), (0.6, 0.2, 0.5, 1.0), (0.4, 0.3, 1.0, 0.0)],
)
def test_zi_initialization_uses_filter_coefficients(alpha, beta, gamma, delta):
    b = [beta * delta, beta * gamma]
    a = [1.0, -alpha]
    expected = lfilter(b, a, FLOW, zi=lfilter_zi(b, a))[0]
    actual, count = general_form_digital_filter(
        FLOW, alpha, beta, gamma, delta=delta, b0_method="zi"
    )
    assert actual == pytest.approx(np.clip(expected, 0, FLOW))
    assert count == np.count_nonzero(expected > FLOW)


@pytest.mark.parametrize(
    ("alpha", "beta", "gamma"), [(None, 0.1, 1), (0.8, None, 1), (0.8, 0.1, None)]
)
def test_zi_requires_filter_coefficients(alpha, beta, gamma):
    with pytest.raises(ValueError, match="requires alpha, beta and gamma"):
        _init_baseflow(FLOW, "zi", alpha=alpha, beta=beta, gamma=gamma)


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_method_rejects_none_initialization(method, kwargs):
    with pytest.raises(ValueError, match="cannot be None"):
        method(FLOW, k=K, b0_method=None, **kwargs)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [(method, kwargs) for method, kwargs in METHODS if method not in (furey, ewma)],
)
def test_zi_is_the_explicit_default(method, kwargs):
    assert method(FLOW, k=K, **kwargs) == pytest.approx(
        method(FLOW, k=K, b0_method="zi", **kwargs)
    )


@pytest.mark.parametrize(
    ("method", "kwargs"), [(ukih, {}), (usgs_hysep_local, {"num_days": 1})]
)
def test_graphical_methods_reject_none(method, kwargs):
    q = pd.read_csv("tests/data_short.csv", index_col=0).iloc[:200, 0].to_numpy()
    with pytest.raises(ValueError, match="cannot be None"):
        method(q, k=K, b0_method=None, **kwargs)


def test_zi_for_furey_and_ewma_uses_their_own_coefficients():
    c3c1, e = 0.4, 0.2
    beta = c3c1 * (1 - K)
    alpha = K - beta
    furey_initial = lfilter_zi([0, beta], [1, -alpha])[0]
    ewma_initial = lfilter_zi([e, 0], [1, -(1 - e)])[0] + e * FLOW[0]
    assert furey(FLOW, K, c3c1, b0_method="zi")[0] == pytest.approx(furey_initial)
    assert ewma(FLOW, K, e, b0_method="zi")[0] == pytest.approx(ewma_initial)


@pytest.mark.parametrize(
    ("name", "candidate_parameters"),
    [
        ("boughton", [0.2, 0.4]),
        ("furey", [0.2, 0.4]),
        ("ewma", [0.2, 0.4]),
    ],
)
def test_zi_calibration_uses_each_candidates_coefficients(
    monkeypatch, name, candidate_parameters
):
    module = importlib.import_module(f"hydrotoolbox.baseflow.methods.{name}")
    initializer_module = (
        importlib.import_module("hydrotoolbox.baseflow.general_form")
        if name == "boughton"
        else module
    )
    original = initializer_module._init_baseflow
    calls = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(initializer_module, "_init_baseflow", record)
    q = np.tile(np.linspace(5, 1, 30), 2)
    param_calibrate(
        np.array(candidate_parameters), getattr(module, f"_{name}"), q, K, "zi"
    )

    assert len(calls) == len(candidate_parameters)
    for candidate, (coefficients, actual) in zip(candidate_parameters, calls):
        if name == "boughton":
            alpha, beta, gamma, delta = (
                K / (1 + candidate),
                candidate / (1 + candidate),
                0,
                1,
            )
        elif name == "furey":
            beta = candidate * (1 - K)
            alpha, gamma, delta = K - beta, 1, 0
        else:
            alpha, beta, gamma, delta = 1 - candidate, candidate, 0, 1
        b = [beta * delta, beta * gamma]
        expected = lfilter_zi(b, [1, -alpha])[0] + b[0] * q[0]
        assert coefficients["alpha"] == pytest.approx(alpha)
        assert coefficients["beta"] == pytest.approx(beta)
        assert actual == pytest.approx(expected)


def test_ewma_legacy_initialization_1():
    assert ewma(FLOW, k=K, e=0.2) == pytest.approx(
        ewma(FLOW, k=K, e=0.2, b0_method=lyne_hollick(FLOW, K)[0])
    )


def test_ewma_legacy_initialization_2():
    assert ewma(FLOW, k=K, e=0.2) == pytest.approx(
        ewma(FLOW, k=K, e=0.2, b0_method="LH")
    )


@pytest.mark.parametrize("initial", ["zi", "Q0", "min", "LH", 1.5, "1.5"])
@pytest.mark.parametrize("passes", [2, 3, 4])
def test_lyne_hollick_initializes_only_first_pass(initial, passes):
    first = lyne_hollick(FLOW, K, passes=1, b0_method=initial)
    expected = first
    for _ in range(1, passes):
        expected = lyne_hollick(expected[::-1], K, passes=1, b0_method="Q0")
    if passes % 2 == 0:
        expected = expected[::-1]
    actual = lyne_hollick(FLOW, K, passes=passes, b0_method=initial)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("name", ["ukih", "usgs_hysep_local"])
def test_graphical_initialization_changes_only_lyne_hollick_edges(name):
    q = pd.read_csv("tests/data_short.csv", index_col=0).iloc[:200, 0].to_numpy()
    if name == "ukih":
        block_end = q.size // 5 * 5
        minima = np.argmin(q[:block_end].reshape(-1, 5), axis=1)
        turns = ukih_turn(q, minima + np.arange(0, block_end, 5))
        func = ukih
    else:
        turns = local_turn(q, hysep_interval(num_days=1))
        func = lambda flow, **kw: usgs_hysep_local(flow, num_days=1, **kw)
    original = func(q, k=K)
    actual = func(q, k=K, b0_method="min")
    edge = lh_filter(q, K, b0_method="min")
    assert actual[: turns[0]] == pytest.approx(edge[: turns[0]])
    assert actual[turns[-1] + 1 :] == pytest.approx(edge[turns[-1] + 1 :])
    assert actual[turns[0] : turns[-1] + 1] == pytest.approx(
        original[turns[0] : turns[-1] + 1]
    )
    assert actual[0] != pytest.approx(original[0])


def test_separation_passes_initialization_to_methods():
    q = pd.read_csv("tests/data_short.csv", index_col=0).iloc[:200, 0].to_numpy()
    values, scores = separation(
        q, method=["chapman", "furey"], k=K, c3c1=0.4, b0_method="min"
    )
    assert values["chapman"] == pytest.approx(chapman(q, k=K, b0_method="min"))
    assert values["furey"] == pytest.approx(furey(q, k=K, c3c1=0.4, b0_method="min"))
    assert scores.shape == (2,)


def test_separation_rejects_none_initialization():
    q = pd.read_csv("tests/data_short.csv", index_col=0).iloc[:200, 0].to_numpy()
    with pytest.raises(ValueError, match="cannot be None"):
        separation(q, method="chapman", k=K, b0_method=None)


@pytest.mark.parametrize(
    ("name", "extra_args"),
    [("ukih", []), ("usgs_hysep_local", ["--num_days=1"])],
)
def test_graphical_api_and_cli_initialization(tmp_path, name, extra_args):
    q = pd.read_csv("tests/data_short.csv", index_col=0).iloc[:200]
    path = tmp_path / "flow.csv"
    q.to_csv(path, index_label="Datetime")
    kwargs = {"num_days": 1} if name == "usgs_hysep_local" else {}
    expected = getattr(baseflow_sep, name)(
        input_ts=str(path), b0_method="min", **kwargs
    )
    result = subprocess.run(
        [
            "hydrotoolbox",
            "baseflow_sep",
            name,
            f"--input_ts={path}",
            "--b0_method=min",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    actual = pd.read_csv(StringIO(result.stdout), index_col=0)
    actual.index = pd.to_datetime(actual.index)
    actual.index.freq = "D"
    assert_series_equal(actual.iloc[:, 0], expected.iloc[:, 0], check_dtype=False)


@pytest.mark.parametrize(
    "name",
    [
        "boughton",
        "chapman",
        "chapman_maxwell",
        "eckhardt",
        "ewma",
        "furey",
        "ihacres",
        "lyne_hollick",
        "ukih",
        "usgs_hysep_local",
        "willems",
    ],
)
def test_zi_public_api_and_cli(tmp_path, name):
    q = pd.read_csv("tests/data_short.csv", index_col=0).iloc[:200]
    path = tmp_path / "flow.csv"
    q.to_csv(path, index_label="Datetime")
    expected = getattr(baseflow_sep, name)(input_ts=str(path), b0_method="zi")
    result = subprocess.run(
        [
            "hydrotoolbox",
            "baseflow_sep",
            name,
            f"--input_ts={path}",
            "--b0_method=zi",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    actual = pd.read_csv(StringIO(result.stdout), index_col=0)
    actual.index = pd.to_datetime(actual.index)
    actual.index.freq = "D"
    assert_series_equal(actual.iloc[:, 0], expected.iloc[:, 0], check_dtype=False)


@pytest.mark.parametrize(
    ("name", "parameter", "extra"),
    [
        ("boughton", "C", {}),
        ("eckhardt", "bfi_max", {}),
        ("ewma", "e", {}),
        ("furey", "c3c1", {}),
        ("ihacres", "C", {"a": 0.5}),
        ("willems", "w", {}),
    ],
)
@pytest.mark.parametrize("initial", ["min", "LH", "zi"])
def test_calibration_propagates_initialization(
    monkeypatch, name, parameter, extra, initial
):
    module = importlib.import_module(f"hydrotoolbox.baseflow.methods.{name}")
    selected = 0.3

    def calibrate(candidates, method, flow, k, b0_method):
        assert b0_method == initial
        calculated, _ = method(flow, k, selected, b0_method=b0_method)
        expected = getattr(module, name)(
            flow, k=k, b0_method=initial, **extra, **{parameter: selected}
        )
        assert calculated == pytest.approx(expected)
        return selected

    monkeypatch.setattr(module, "param_calibrate", calibrate)
    actual = getattr(module, name)(FLOW, k=K, b0_method=initial, **extra)
    expected = getattr(module, name)(
        FLOW, k=K, b0_method=initial, **extra, **{parameter: selected}
    )
    assert actual == pytest.approx(expected)


def test_param_calibrate_initializes_once(monkeypatch):
    module = importlib.import_module("hydrotoolbox.baseflow.param_estimate")
    calls = []

    def init(flow, initial, k):
        calls.append((initial, k))
        return 1.5

    def method(flow, k, candidate, b0_method):
        assert b0_method == 1.5
        return flow * candidate, 0

    monkeypatch.setattr(module, "_init_baseflow", init)
    q = np.tile(np.linspace(5, 1, 30), 2)
    param_calibrate(np.array([0.3, 0.5, 0.7]), method, q, K, b0_method="LH")
    assert calls == [("LH", K)]


def test_param_calibrate_passes_zi_to_each_candidate():
    calls = []

    def method(flow, k, candidate, b0_method):
        calls.append((candidate, b0_method))
        return flow * candidate, 0

    q = np.tile(np.linspace(5, 1, 30), 2)
    param_calibrate(np.array([0.3, 0.5, 0.7]), method, q, K, b0_method="zi")
    assert calls == [(0.3, "zi"), (0.5, "zi"), (0.7, "zi")]


@pytest.mark.parametrize(
    ("name", "kwargs", "args"),
    [
        ("furey", {"k": K, "c3c1": 0.4}, ["-k=0.8", "--c3c1=0.4"]),
        ("chapman", {"k": K}, ["-k=0.8"]),
        ("lyne_hollick", {"alpha": K}, ["--alpha=0.8"]),
    ],
)
@pytest.mark.parametrize("initial", ["min", "1.5", "LH", "zi"])
def test_api_and_cli_initialization(tmp_path, name, kwargs, args, initial):
    path = tmp_path / "flow.csv"
    pd.DataFrame(
        {"flow": FLOW}, index=pd.date_range("2020-01-01", periods=len(FLOW))
    ).to_csv(path, index_label="Datetime")
    expected = getattr(baseflow_sep, name)(
        input_ts=str(path), b0_method=initial, **kwargs
    )
    command = [
        "hydrotoolbox",
        "baseflow_sep",
        name,
        f"--input_ts={path}",
        f"--b0_method={initial}",
        *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    actual = pd.read_csv(StringIO(result.stdout), index_col=0)
    actual.index = pd.to_datetime(actual.index)
    actual.index.freq = "D"
    assert_series_equal(actual.iloc[:, 0], expected.iloc[:, 0], check_dtype=False)
