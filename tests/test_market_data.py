"""
EVA-AI-01
=========
Market Data Tests - Phase 3

These tests verify the research-data boundary without contacting Binance.

Coverage
--------
- OHLC integrity
- NaN / Inf rejection
- volume integrity
- quote-volume integrity
- trade-count integrity
- timestamp integrity
- candle continuity
- duplicate detection
- missing interval detection
- symbol validation
- interval validation
- limit validation
- timeout validation
- Binance kline parsing
- incomplete-candle filtering
- malformed response handling
"""

from __future__ import annotations

from datetime import datetime, timezone
import math

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
    """Create one deterministic test candle."""

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
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
    )

    assert candle.is_valid_ohlc is True


def test_invalid_ohlc_is_rejected() -> None:
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        close=110.0,
    )

    assert candle.is_valid_ohlc is False

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_infinite_ohlc_is_rejected() -> None:
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        high=math.inf,
    )

    assert candle.is_valid_ohlc is False

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_nan_ohlc_is_rejected() -> None:
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        close=math.nan,
    )

    assert candle.is_valid_ohlc is False

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_negative_volume_is_rejected() -> None:
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        volume=-1.0,
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_infinite_volume_is_rejected() -> None:
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        volume=math.inf,
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_negative_quote_volume_is_rejected() -> None:
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        quote_volume=-1.0,
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_infinite_quote_volume_is_rejected() -> None:
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        quote_volume=math.inf,
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_negative_trade_count_is_rejected() -> None:
    candle = make_candle(
        open_time=1_000,
        close_time=59_999,
        trades=-1,
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_invalid_timestamp_order_is_rejected() -> None:
    candle = make_candle(
        open_time=60_000,
        close_time=59_999,
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (candle,),
            interval="1m",
        )


def test_duplicate_timestamp_is_rejected() -> None:
    candles = (
        make_candle(
            1_000,
            59_999,
        ),
        make_candle(
            1_000,
            59_999,
        ),
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            candles,
            interval="1m",
        )


def test_non_monotonic_timestamps_are_rejected() -> None:
    candles = (
        make_candle(
            61_000,
            119_999,
        ),
        make_candle(
            1_000,
            59_999,
        ),
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            candles,
            interval="1m",
        )


def test_missing_interval_is_rejected() -> None:
    candles = (
        make_candle(
            1_000,
            59_999,
        ),
        make_candle(
            121_000,
            179_999,
        ),
    )

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            candles,
            interval="1m",
        )


def test_valid_continuous_candles_are_accepted() -> None:
    candles = (
        make_candle(
            1_000,
            59_999,
        ),
        make_candle(
            61_000,
            119_999,
        ),
        make_candle(
            121_000,
            179_999,
        ),
    )

    BinanceMarketDataClient._validate_candles(
        candles,
        interval="1m",
    )


def test_empty_candle_sequence_is_rejected() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_candles(
            (),
            interval="1m",
        )


def test_invalid_symbol_is_rejected() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_symbol(
            "BTC/USDT"
        )


def test_empty_symbol_is_rejected() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_symbol(
            ""
        )


def test_symbol_is_normalized() -> None:
    result = (
        BinanceMarketDataClient
        ._validate_symbol(
            " btcusdt "
        )
    )

    assert result == "BTCUSDT"


def test_invalid_interval_is_rejected() -> None:
    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._validate_interval(
            "13m"
        )


def test_interval_is_normalized() -> None:
    result = (
        BinanceMarketDataClient
        ._validate_interval(
            " 1m "
        )
    )

    assert result == "1m"


def test_invalid_limit_is_rejected() -> None:
    client = BinanceMarketDataClient()

    with pytest.raises(
        MarketDataValidationError
    ):
        client.get_klines(
            symbol="BTCUSDT",
            interval="1m",
            limit=0,
        )


def test_limit_above_binance_maximum_is_rejected() -> None:
    client = BinanceMarketDataClient()

    with pytest.raises(
        MarketDataValidationError
    ):
        client.get_klines(
            symbol="BTCUSDT",
            interval="1m",
            limit=1001,
        )


def test_boolean_limit_is_rejected() -> None:
    client = BinanceMarketDataClient()

    with pytest.raises(
        MarketDataValidationError
    ):
        client.get_klines(
            symbol="BTCUSDT",
            interval="1m",
            limit=True,
        )


def test_invalid_timeout_is_rejected() -> None:
    for value in (
        0,
        -1,
        math.inf,
        math.nan,
        True,
        "invalid",
    ):
        with pytest.raises(
            MarketDataValidationError
        ):
            BinanceMarketDataClient(
                timeout=value,
            )


def test_invalid_end_time_is_rejected() -> None:
    client = BinanceMarketDataClient()

    with pytest.raises(
        MarketDataValidationError
    ):
        client.get_klines(
            symbol="BTCUSDT",
            interval="1m",
            end_time=True,
        )


def test_binance_kline_row_is_parsed() -> None:
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

    candle = (
        BinanceMarketDataClient
        ._parse_candle(row)
    )

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
    row = [
        1_000,
        "100.0",
        "105.0",
        "99.0",
    ]

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._parse_candle(
            row
        )


def test_invalid_numeric_kline_row_is_rejected() -> None:
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

    with pytest.raises(
        MarketDataValidationError
    ):
        BinanceMarketDataClient._parse_candle(
            row
        )


def test_candle_datetime_is_utc() -> None:
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
    future_close_time = (
        int(
            datetime.now(
                timezone.utc
            ).timestamp()
            * 1000
        )
        + 60_000
    )

    candle = make_candle(
        open_time=future_close_time - 59_999,
        close_time=future_close_time,
    )

    assert (
        BinanceMarketDataClient
        ._is_completed(candle)
        is False
    )


def test_completed_candle_is_completed() -> None:
    past_close_time = (
        int(
            datetime.now(
                timezone.utc
            ).timestamp()
            * 1000
        )
        - 60_000
    )

    candle = make_candle(
        open_time=past_close_time - 59_999,
        close_time=past_close_time,
    )

    assert (
        BinanceMarketDataClient
        ._is_completed(candle)
        is True
    )


def test_interval_conversion() -> None:
    assert (
        BinanceMarketDataClient
        ._interval_to_milliseconds("1m")
        == 60_000
    )

    assert (
        BinanceMarketDataClient
        ._interval_to_milliseconds("1h")
        == 3_600_000
    )

    assert (
        BinanceMarketDataClient
        ._interval_to_milliseconds("1d")
        == 86_400_000
    )


def test_month_interval_has_no_fixed_duration() -> None:
    assert (
        BinanceMarketDataClient
        ._interval_to_milliseconds("1M")
        is None
    )


def test_public_market_data_base_url() -> None:
    client = BinanceMarketDataClient()

    assert (
        client.base_url
        == "https://data-api.binance.vision"
    )
