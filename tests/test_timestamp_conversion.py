from __future__ import annotations

import pandas as pd
import pytest

from core.system_integration import (
    IntegrationError,
    _candles_to_frame,
)
from data.market_data import Candle


def _candle(open_time: int) -> Candle:
    return Candle(
        open_time=open_time,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=10.0,
        close_time=open_time + 59_999,
        quote_volume=1_020.0,
        number_of_trades=20,
    )


def test_candle_timestamp_is_converted_from_milliseconds_to_utc():
    timestamp_ms = 1_700_000_000_000

    frame = _candles_to_frame(
        (
            _candle(timestamp_ms),
            _candle(timestamp_ms + 60_000),
        )
    )

    assert pd.api.types.is_datetime64_any_dtype(
        frame["timestamp"]
    )

    assert str(frame["timestamp"].dt.tz) == "UTC"

    assert frame["timestamp"].iloc[0] == pd.Timestamp(
        "2023-11-14T22:13:20Z"
    )

    assert frame["timestamp"].iloc[1] == pd.Timestamp(
        "2023-11-14T22:14:20Z"
    )


def test_candle_timestamps_preserve_one_minute_spacing():
    timestamp_ms = 1_700_000_000_000

    frame = _candles_to_frame(
        (
            _candle(timestamp_ms),
            _candle(timestamp_ms + 60_000),
            _candle(timestamp_ms + 120_000),
        )
    )

    differences = frame["timestamp"].diff().dropna()

    assert all(
        difference == pd.Timedelta(minutes=1)
        for difference in differences
    )


def test_candle_timestamp_is_not_interpreted_as_nanoseconds():
    timestamp_ms = 1_700_000_000_000

    frame = _candles_to_frame(
        (
            _candle(timestamp_ms),
            _candle(timestamp_ms + 60_000),
        )
    )

    first_timestamp = frame["timestamp"].iloc[0]

    assert first_timestamp.year == 2023
    assert first_timestamp.year != 1970


def test_invalid_market_data_type_is_rejected():
    from core.system_integration import _validate_market_data_result

    with pytest.raises(IntegrationError):
        _validate_market_data_result(None)
