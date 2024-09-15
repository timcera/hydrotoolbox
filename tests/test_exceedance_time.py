import pandas as pd
import pytest

from hydrotoolbox import hydrotoolbox


@pytest.mark.parametrize(
    "thresholds, input_ts, delays, under_over, time_units, columns, source_units, start_date, end_date, dropna, clean, round_index, skiprows, index_type, names, target_units, expected",
    [
        # Test case 1: Happy path, over threshold, day units
        (
            [10, 20],
            pd.Series(
                [5, 15, 25, 35],
                index=pd.date_range(start="1/1/2022", periods=4, freq="D"),
            ),
            [0],
            "over",
            "day",
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
            {10: 2.5, 20: 1.5},
        ),
        # Test case 2: Happy path, under threshold, hour units
        (
            [30, 40],
            pd.Series(
                [5, 15, 25, 35],
                index=pd.date_range(start="1/1/2022", periods=4, freq="H"),
            ),
            [0],
            "under",
            "hour",
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
            {30: 2.5, 40: 3},
        ),
        # Test case 3: Edge case, single threshold, month units
        (
            [15],
            pd.Series(
                [5, 15, 25, 35],
                index=pd.date_range(start="1/1/2022", periods=4, freq="M"),
            ),
            [0],
            "over",
            "month",
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
            {15: 2.0040719118222046},
        ),
        # Test case 4: Error case, mismatched thresholds and delays
        (
            [10, 20],
            pd.Series(
                [5, 15, 25, 35], index=pd.date_range(start="1/1/2022", periods=4)
            ),
            [1],
            "over",
            "day",
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
            ValueError,
        ),
    ],
    ids=[
        "happy_path_over_day",
        "happy_path_under_hour",
        "edge_case_single_threshold",
        "error_case_mismatched_thresholds_delays",
    ],
)
def test_exceedance_time(
    thresholds,
    input_ts,
    delays,
    under_over,
    time_units,
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
    if expected is ValueError:
        # Assert
        with pytest.raises(ValueError):
            # Act
            hydrotoolbox.exceedance_time(
                *thresholds,
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
            )
    else:
        # Act
        result = hydrotoolbox.exceedance_time(
            *thresholds,
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
        )
        # Assert
        assert result == expected
