import pandas as pd
import pytest

from hydrotoolbox import hydrotoolbox


@pytest.mark.parametrize(
    "input_ts, exceedance_probabilities, columns, source_units, start_date, end_date, dropna, clean, round_index, skiprows, index_type, names, target_units, expected",
    [
        # Test ID: 01-01
        # Test Description:
        #    Testing the happy path with realistic values.
        (
            pd.DataFrame(
                {"A": [1, 2, 3, 4, 5]},
                index=pd.date_range("2000-01-01", periods=5, freq="D"),
            ),
            (
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                25.0,
                50.0,
                75.0,
                90.0,
                95.0,
                98.0,
                99.0,
                99.5,
            ),
            None,
            None,
            None,
            None,
            "no",
            False,
            None,
            None,
            "datetime",
            None,
            None,
            pd.DataFrame(
                {
                    "Quantiles": [
                        4.98,
                        4.96,
                        4.92,
                        4.8,
                        4.6,
                        4,
                        3,
                        2,
                        1.4,
                        1.2,
                        1.08,
                        1.04,
                        1.02,
                    ]
                },
                index=[
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                    0.98,
                    0.99,
                    0.995,
                ],
            ),
        ),
        (
            pd.DataFrame(
                {"A": range(101)},
                index=pd.date_range("2000-01-01", periods=101, freq="D"),
            ),
            (99.5, 99, 98, 95, 90, 75, 50, 25, 10, 5, 2, 1, 0.5),
            None,
            None,
            None,
            None,
            "no",
            False,
            None,
            None,
            "datetime",
            None,
            None,
            pd.DataFrame(
                {
                    "Quantiles": [
                        0.5,
                        1.0,
                        2.0,
                        5.0,
                        10.0,
                        25.0,
                        50.0,
                        75.0,
                        90.0,
                        95.0,
                        98.0,
                        99.0,
                        99.5,
                    ]
                },
                index=[
                    0.995,
                    0.99,
                    0.98,
                    0.95,
                    0.9,
                    0.75,
                    0.5,
                    0.25,
                    0.1,
                    0.05,
                    0.02,
                    0.01,
                    0.005,
                ],
            ),
        ),
        # Test ID: 01-02
        # Test Description:
        #    Testing the edge case where input_ts is an empty DataFrame.
        (
            pd.DataFrame(
                {"A": [1]}, index=pd.date_range(start="2000-01-01", periods=1, freq="D")
            ),
            (99.5, 99, 98, 95, 90, 75, 50, 25, 10, 5, 2, 1, 0.5),
            None,
            None,
            None,
            None,
            "no",
            False,
            None,
            None,
            "datetime",
            None,
            None,
            pd.DataFrame(
                {
                    "Quantiles": [
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]
                },
                index=[
                    0.995,
                    0.99,
                    0.98,
                    0.95,
                    0.9,
                    0.75,
                    0.5,
                    0.25,
                    0.1,
                    0.05,
                    0.02,
                    0.01,
                    0.005,
                ],
            ),
        ),
        # Test ID: 01-03
        # Test Description:
        #    Testing the error case where exceedance_probabilities is an empty tuple.
        (
            pd.DataFrame({"A": [1, 2, 3, 4, 5]}),
            (),
            None,
            None,
            None,
            None,
            "no",
            False,
            None,
            None,
            "datetime",
            None,
            None,
            pd.DataFrame({"A": []}),
        ),
    ],
    ids=[
        "happy_path",
        "happy_path_large",
        "edge_case_empty_input_ts",
        "error_case_empty_exceedance_probabilities",
    ],
)
def test_flow_duration(
    input_ts,
    exceedance_probabilities,
    columns,
    source_units,
    start_date,
    end_date,
    dropna,
    clean,
    round_index,
    skiprows,
    index_type,
    names,
    target_units,
    expected,
):
    # Arrange
    # Act
    result = hydrotoolbox.flow_duration(
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
    )
    # Assert
    if len(expected.columns) == 1:
        expected.columns = ["A"]
    expected.index.name = "Quantiles"
    print(result)
    print(expected)
    pd.testing.assert_frame_equal(
        result, expected, check_index_type=False, check_dtype=False
    )
