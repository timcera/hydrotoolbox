import pytest

from hydrotoolbox.hydrotoolbox import recession


@pytest.mark.parametrize(
    "input_ts, columns, source_units, start_date, end_date, dropna, clean, round_index, skiprows, index_type, names, target_units, expected",
    [
        # Test ID: 1-1
        # Test Description:
        # Happy path test with realistic values.
        (
            "tests/data.csv",
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
            {"Q": [0.9692298918928195]},
        ),
        # Test ID: 1-2
        # Test Description:
        # Edge case with empty input time series.
        (
            "",
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
            TypeError,
        ),
    ],
    ids=["happy_path", "edge_empty_input"],
)
def test_recession(
    input_ts,
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
    if issubclass(type(expected), type(Exception)):
        with pytest.raises(expected):
            result = recession(
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
    else:
        # Act
        result = recession(
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

        # Assert
        assert result == expected
