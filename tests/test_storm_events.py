import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hydrotoolbox import storm_events


@pytest.mark.parametrize(
    "rise_lag, fall_lag, input_ts, min_peak, window, columns, source_units, start_date, end_date, dropna, clean, round_index, skiprows, index_type, names, target_units, expected",
    [
        # Test ID: 01-01
        # Test Description:
        # Testing with realistic values and expecting a DataFrame with peak storm events.
        (
            2,
            2,
            pd.DataFrame(
                {
                    "Datetime": pd.date_range(start="1/1/2020", periods=10, freq="D"),
                    "flow": [1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
                }
            ).set_index("Datetime"),
            None,
            1,
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
                    "Datetime": pd.DatetimeIndex(
                        pd.date_range(start="1/3/2020", periods=5, freq="D")
                    ),
                    "flow": [3, 4, 5, 4, 3],
                }
            ).set_index("Datetime"),
        ),
        # Test ID: 01-02
        # Test Description:
        # Testing with edge case where input_ts is empty and expecting an empty DataFrame.
        (
            2,
            2,
            pd.DataFrame(
                {
                    "Datetime": pd.date_range(start="1/1/2020", periods=0, freq="D"),
                    "flow": [],
                }
            ).set_index("Datetime"),
            None,
            1,
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
        # Test ID: 01-03
        # Test Description:
        # Testing with error case where rise_lag is negative and expecting a ValueError.
        (
            -2,
            2,
            pd.DataFrame(
                {
                    "Datetime": pd.date_range(start="1/1/2020", periods=10, freq="D"),
                    "flow": [1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
                }
            ).set_index("Datetime"),
            None,
            1,
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
    ids=["01-01", "01-02", "01-03"],
)
def test_storm_events(
    rise_lag,
    fall_lag,
    input_ts,
    min_peak,
    window,
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
    if isinstance(expected, type) and issubclass(expected, Exception):
        # Act and Assert
        with pytest.raises(expected):
            storm_events(
                rise_lag,
                fall_lag,
                input_ts,
                min_peak,
                window,
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
            )
    else:
        # Act
        result = storm_events(
            rise_lag,
            fall_lag,
            input_ts,
            min_peak,
            window,
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
        )

        # Assert
        assert_frame_equal(
            result,
            expected,
            check_index_type=False,
            check_freq=False,
            check_dtype=False,
        )
