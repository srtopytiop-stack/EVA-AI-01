"""
EVA-AI-01
=========
Market Data Tests - Phase 3

Tests:
- OHLC integrity
- Timestamp integrity
- Candle continuity
- Duplicate detection
- Missing interval detection
- Symbol validation
- Interval validation
- Limit validation
- Binance kline parsing
- Incomplete candle filtering
- Malformed response handling

These tests do not place orders.
These tests do not require API keys.
Network access is not required.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.market_data import (
    BinanceMarketDataClient,
    Candle,
    MarketDataValidationError,
)


def make_candle(
    open_time: int,
    close_time: int,
    *,
    open_price: float = 100.0,
    high: float = 105.0,
    low: float = 99.0,
    close: float = 103.0,
    volume: float = 10.0,
    quote_volume: float = 1000.0,
    trades: int = 100,
) -> Candle:
    """Create a deterministic test candle."""

    return Candle(
        open_time=open_time,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_time=close_time,
        quote_volume=quote_volume,
        number_of_trades=trades,
    )


def test_valid_ohlc_is_accepted() -> None:
    """A structurally valid OHLC candle must pass validation."""

    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
    )

    assert candle.is_valid_ohlc is True


def test_invalid_ohlc_is_rejected() -> None:
    """A candle where close exceeds high must be rejected."""

    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        close=110.0,
    )

    assert candle.is_valid_ohlc is False

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_negative_volume_is_rejected() -> None:
    """Negative volume is impossible and must fail validation."""

    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        volume=-1.0,
    )

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_negative_quote_volume_is_rejected() -> None:
    """Negative quote volume must fail validation."""

    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        quote_volume=-1.0,
    )

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_negative_trade_count_is_rejected() -> None:
    """Negative trade count must fail validation."""

    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        trades=-1,
    )

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_invalid_timestamp_order_is_rejected() -> None:
    """Close time cannot precede open time."""

    candle = make_candle(
        open_time=60_000,
        close_time=59_999,
    )

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_duplicate_timestamp_is_rejected() -> None:
    """Two candles cannot have the same opening timestamp."""

    candles = (
        make_candle(1_000, 59_999),
        make_candle(1_000, 59_999),
    )

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            candles,
            interval="1m",
        )


def test_non_monotonic_timestamps_are_rejected() -> None:
    """Candle timestamps must increase strictly."""

    candles = (
        make_candle(61_000, 119_999),
        make_candle(1_000, 59_999),
    )

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            candles,
            interval="1m",
        )


def test_missing_interval_is_rejected() -> None:
    """A missing 1-minute candle must be detected."""

    candles = (
        make_candle(1_000, 59_999),
        make_candle(121_000, 179_999),
    )

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            candles,
            interval="1m",
        )


def test_valid_continuous_candles_are_accepted() -> None:
    """Continuous candles with correct spacing must pass."""

    candles = (
        make_candle(1_000, 59_999),
        make_candle(61_000, 119_999),
        make_candle(121_000, 179_999),
    )

    BinanceMarketDataClient._validate_candles(
        candles,
        interval="1m",
    )


def test_empty_candle_sequence_is_rejected() -> None:
    """An empty market-data response must fail validation."""

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_candles(
            (),
            interval="1m",
        )


def test_invalid_symbol_is_rejected() -> None:
    """Symbols containing invalid characters must fail."""

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_symbol(
            "BTC/USDT"
        )


def test_empty_symbol_is_rejected() -> None:
    """An empty symbol must fail."""

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_symbol("")


def test_symbol_is_normalized() -> None:
    """Valid symbols should be normalized to uppercase."""

    result = BinanceMarketDataClient._validate_symbol(
        " btcusdt "
    )

    assert result == "BTCUSDT"


def test_invalid_interval_is_rejected() -> None:
    """Unsupported Binance intervals must fail."""

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._validate_interval(
            "13m"
        )


def test_interval_is_normalized() -> None:
    """Whitespace around a valid interval should be removed."""

    result = BinanceMarketDataClient._validate_interval(
        " 1m "
    )

    assert result == "1m"


def test_invalid_limit_is_rejected() -> None:
    """The Binance limit must remain within 1..1000."""

    client = BinanceMarketDataClient()

    with pytest.raises(MarketDataValidationError):
        client.get_klines(
            symbol="BTCUSDT",
            interval="1m",
            limit=0,
        )


def test_limit_above_binance_maximum_is_rejected() -> None:
    """The client must reject limits above 1000."""

    client = BinanceMarketDataClient()

    with pytest.raises(MarketDataValidationError):
        client.get_klines(
            symbol="BTCUSDT",
            interval="1m",
            limit=1001,
        )


def test_binance_kline_row_is_parsed() -> None:
    """A valid Binance kline row must become a Candle."""

    row = [
        1_000,
        "100.0",
        "105.0",
        "99.0",
        "103.0",
        "10.0",
        59_999,
        "1030.0",
        100,
        "500.0",
        "50000.0",
        "0",
    ]

    candle = BinanceMarketDataClient._parse_candle(row)

    assert candle.open_time == 1_000
    assert candle.open == 100.0
    assert candle.high == 105.0
    assert candle.low == 99.0
    assert candle.close == 103.0
    assert candle.volume == 10.0
    assert candle.close_time == 59_999
    assert candle.quote_volume == 1030.0
    assert candle.number_of_trades == 100


def test_malformed_kline_row_is_rejected() -> None:
    """A Binance row with insufficient fields must fail."""

    row = [
        1_000,
        "100.0",
        "105.0",
    ]

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._parse_candle(row)


def test_invalid_numeric_kline_row_is_rejected() -> None:
    """Non-numeric OHLC data must fail parsing."""

    row = [
        1_000,
        "NOT_A_PRICE",
        "105.0",
        "99.0",
        "103.0",
        "10.0",
        59_999,
        "1030.0",
        100,
        "500.0",
        "50000.0",
        "0",
    ]

    with pytest.raises(MarketDataValidationError):
        BinanceMarketDataClient._parse_candle(row)


def test_candle_datetime_is_utc() -> None:
    """Candle datetime conversion must produce UTC."""

    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
    )

    result = candle.datetime

    assert result.tzinfo == timezone.utc
    assert result == datetime.fromtimestamp(
        1.0,
        tz=timezone.utc,
    )


def test_incomplete_candle_is_not_completed() -> None:
    """A candle whose close time is in the future must not be completed."""

    future_close_time = (
        int(datetime.now(timezone.utc).timestamp() * 1000)
        + 60_000
    )

    candle = make_candle(
        open_time=future_close_time - 59_999,
        close_time=future_close_time,
    )

    assert (
        BinanceMarketDataClient._is_completed(candle)
        is False
    )


def test_completed_candle_is_completed() -> None:
    """A candle whose close time is in the past must be completed."""

    past_close_time = (
        int(datetime.now(timezone.utc).timestamp() * 1000)
        - 60_000
    )

    candle = make_candle(
        open_time=past_close_time - 59_999,
        close_time=past_close_time,
    )

    assert (
        BinanceMarketDataClient._is_completed(candle)
        is True
    )


def test_interval_conversion() -> None:
    """Fixed-duration Binance intervals must convert correctly."""

    assert (
        BinanceMarketDataClient._interval_to_milliseconds("1m")
        == 60_000
    )

    assert (
        BinanceMarketDataClient._interval_to_milliseconds("1h")
        == 3_600_000
    )

    assert (
        BinanceMarketDataClient._interval_to_milliseconds("1d")
        == 86_400_000
    )


def test_month_interval_has_no_fixed_duration() -> None:
    """Calendar months do not have a constant millisecond duration."""

    assert (
        BinanceMarketDataClient._interval_to_milliseconds("1M")
        is None
    )
