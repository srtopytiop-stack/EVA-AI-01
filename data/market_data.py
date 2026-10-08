"""
EVA-AI-01
=========
Market Data Layer - Phase 3

Scientific responsibilities
---------------------------
1. Retrieve public Binance Spot OHLCV data.
2. Normalize exchange responses into immutable Candle records.
3. Reject malformed, non-finite, and physically impossible values.
4. Validate chronological ordering.
5. Detect duplicates and missing fixed-duration intervals.
6. Remove incomplete candles before research consumption.
7. Keep market-data acquisition independent from trading execution.
8. Never use API keys, trading credentials, or order endpoints.

Research safety
---------------
This layer does not:
- generate trading signals;
- select assets;
- calculate forecasts;
- place orders;
- use leverage or margin;
- perform live trading.

All internal timestamps are UTC milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any

import requests


BINANCE_SPOT_BASE_URL = "https://data-api.binance.vision"
KLINES_ENDPOINT = "/api/v3/klines"


class MarketDataError(RuntimeError):
    """Base exception for market-data failures."""


class MarketDataRequestError(MarketDataError):
    """Raised when Binance cannot be queried successfully."""


class MarketDataValidationError(MarketDataError):
    """Raised when market data fails a structural or numerical invariant."""


@dataclass(frozen=True)
class Candle:
    """
    Immutable OHLCV candle.

    All timestamps are Unix milliseconds in UTC.

    Parameters
    ----------
    open_time:
        Candle opening timestamp in milliseconds.

    open, high, low, close:
        Positive finite OHLC prices.

    volume:
        Base-asset volume.

    close_time:
        Candle closing timestamp in milliseconds.

    quote_volume:
        Quote-asset volume.

    number_of_trades:
        Number of trades contained in the candle.
    """

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    number_of_trades: int

    @property
    def datetime(self) -> datetime:
        """Return candle opening time as an aware UTC datetime."""

        return datetime.fromtimestamp(
            self.open_time / 1000,
            tz=timezone.utc,
        )

    @property
    def is_valid_ohlc(self) -> bool:
        """
        Validate the OHLC geometry.

        This property deliberately checks finiteness explicitly.
        NaN/Inf must never pass into downstream research mathematics.
        """

        prices = (
            self.open,
            self.high,
            self.low,
            self.close,
        )

        return (
            all(
                math.isfinite(value)
                for value in prices
            )
            and self.low <= self.open <= self.high
            and self.low <= self.close <= self.high
            and self.low > 0.0
            and self.high > 0.0
            and self.open > 0.0
            and self.close > 0.0
        )


@dataclass(frozen=True)
class MarketDataResult:
    """
    Validated market-data response.

    ``candles`` contains completed candles returned by the acquisition layer.
    """

    symbol: str
    interval: str
    candles: tuple[Candle, ...]

    @property
    def count(self) -> int:
        """Return the number of completed candles."""

        return len(self.candles)

    @property
    def last_candle(self) -> Candle | None:
        """Return the latest completed candle, if available."""

        if not self.candles:
            return None

        return self.candles[-1]


class BinanceMarketDataClient:
    """
    Public Binance Spot market-data client.

    Security boundary
    -----------------
    This class intentionally contains:
    - no API key,
    - no API secret,
    - no order endpoint,
    - no account endpoint,
    - no portfolio state.

    It is therefore suitable for EVA's research/paper-only stage.
    """

    VALID_INTERVALS = {
        "1s",
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    }

    def __init__(
        self,
        base_url: str = BINANCE_SPOT_BASE_URL,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        if (
            not isinstance(base_url, str)
            or not base_url.strip()
        ):
            raise MarketDataValidationError(
                "base_url must be a non-empty string."
            )

        try:
            normalized_timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise MarketDataValidationError(
                "timeout must be numeric."
            ) from exc

        if (
            isinstance(timeout, bool)
            or not math.isfinite(normalized_timeout)
            or normalized_timeout <= 0.0
        ):
            raise MarketDataValidationError(
                "timeout must be finite and > 0."
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = normalized_timeout
        self.session = session or requests.Session()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        end_time: int | None = None,
    ) -> MarketDataResult:
        """
        Retrieve validated Binance Spot candles.

        The returned result excludes any candle that has not fully closed.

        Parameters
        ----------
        symbol:
            Binance Spot symbol such as ``BTCUSDT``.

        interval:
            Binance-supported kline interval.

        limit:
            Requested number of candles. Binance maximum is 1000.

        end_time:
            Optional Unix timestamp in milliseconds.
        """

        normalized_symbol = self._validate_symbol(symbol)
        normalized_interval = self._validate_interval(interval)

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise MarketDataValidationError(
                "limit must be between 1 and 1000."
            )

        params: dict[str, Any] = {
            "symbol": normalized_symbol,
            "interval": normalized_interval,
            "limit": limit,
        }

        if end_time is not None:
            if (
                isinstance(end_time, bool)
                or not isinstance(end_time, int)
                or end_time <= 0
            ):
                raise MarketDataValidationError(
                    "end_time must be a positive Unix timestamp "
                    "in milliseconds."
                )

            params["endTime"] = end_time

        payload = self._request_klines(params)

        candles = tuple(
            self._parse_candle(row)
            for row in payload
        )

        self._validate_candles(
            candles,
            interval=normalized_interval,
        )

        completed = tuple(
            candle
            for candle in candles
            if self._is_completed(candle)
        )

        return MarketDataResult(
            symbol=normalized_symbol,
            interval=normalized_interval,
            candles=completed,
        )

    def _request_klines(
        self,
        params: dict[str, Any],
    ) -> list[list[Any]]:
        """Perform one public Binance kline request."""

        url = (
            f"{self.base_url}"
            f"{KLINES_ENDPOINT}"
        )

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise MarketDataRequestError(
                f"Binance market-data request failed: {exc}"
            ) from exc

        if response.status_code != 200:
            raise MarketDataRequestError(
                "Binance returned HTTP "
                f"{response.status_code}: "
                f"{response.text[:300]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketDataRequestError(
                "Binance returned invalid JSON."
            ) from exc

        if not isinstance(payload, list):
            raise MarketDataRequestError(
                "Unexpected Binance response format."
            )

        return payload

    @staticmethod
    def _parse_candle(
        row: list[Any],
    ) -> Candle:
        """
        Convert one Binance kline row into a Candle.

        Binance's kline response contains more fields than EVA currently
        needs. Only the fields required by the research contract are mapped.
        """

        if len(row) < 11:
            raise MarketDataValidationError(
                "Malformed Binance kline: insufficient fields."
            )

        try:
            return Candle(
                open_time=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=int(row[6]),
                quote_volume=float(row[7]),
                number_of_trades=int(row[8]),
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise MarketDataValidationError(
                "Malformed Binance kline: numeric conversion failed."
            ) from exc

    @classmethod
    def _validate_candles(
        cls,
        candles: tuple[Candle, ...],
        interval: str,
    ) -> None:
        """
        Validate the complete OHLCV sequence.

        Invariants
        ----------
        1. Sequence cannot be empty.
        2. OHLC prices must be finite and positive.
        3. Volume fields must be finite and non-negative.
        4. Trade count must be non-negative.
        5. close_time >= open_time.
        6. open_time must increase strictly.
        7. Fixed-duration intervals must have exact spacing.
        """

        if not candles:
            raise MarketDataValidationError(
                "Binance returned no candles."
            )

        previous_time: int | None = None

        for candle in candles:
            if not candle.is_valid_ohlc:
                raise MarketDataValidationError(
                    "Invalid OHLC relationship at "
                    f"{candle.open_time}."
                )

            if (
                not math.isfinite(candle.volume)
                or candle.volume < 0.0
            ):
                raise MarketDataValidationError(
                    f"Invalid volume at {candle.open_time}."
                )

            if (
                not math.isfinite(candle.quote_volume)
                or candle.quote_volume < 0.0
            ):
                raise MarketDataValidationError(
                    f"Invalid quote volume at "
                    f"{candle.open_time}."
                )

            if candle.number_of_trades < 0:
                raise MarketDataValidationError(
                    f"Negative trade count at "
                    f"{candle.open_time}."
                )

            if candle.close_time < candle.open_time:
                raise MarketDataValidationError(
                    "Invalid candle timestamps at "
                    f"{candle.open_time}."
                )

            if (
                previous_time is not None
                and candle.open_time <= previous_time
            ):
                raise MarketDataValidationError(
                    "Candle timestamps are not "
                    "strictly increasing."
                )

            previous_time = candle.open_time

        interval_ms = cls._interval_to_milliseconds(
            interval
        )

        # Calendar-month candles have no fixed millisecond
        # duration, therefore continuity cannot be checked using
        # a constant delta.
        if interval_ms is None:
            return

        for previous, current in zip(
            candles,
            candles[1:],
        ):
            difference = (
                current.open_time
                - previous.open_time
            )

            if difference != interval_ms:
                raise MarketDataValidationError(
                    "Missing or irregular candle interval "
                    "detected: "
                    f"{previous.open_time} -> "
                    f"{current.open_time}."
                )

    @staticmethod
    def _is_completed(
        candle: Candle,
    ) -> bool:
        """
        Determine whether a candle has fully closed.

        A forming final candle must not enter research calculations.
        """

        now_ms = int(
            datetime.now(
                timezone.utc
            ).timestamp()
            * 1000
        )

        return candle.close_time < now_ms

    @classmethod
    def _validate_symbol(
        cls,
        symbol: str,
    ) -> str:
        """Validate and normalize a Spot symbol."""

        if not isinstance(symbol, str):
            raise MarketDataValidationError(
                "symbol must be a string."
            )

        normalized = symbol.strip().upper()

        if not normalized:
            raise MarketDataValidationError(
                "symbol cannot be empty."
            )

        if not normalized.isalnum():
            raise MarketDataValidationError(
                "symbol contains invalid characters."
            )

        return normalized

    @classmethod
    def _validate_interval(
        cls,
        interval: str,
    ) -> str:
        """Validate a Binance kline interval."""

        if not isinstance(interval, str):
            raise MarketDataValidationError(
                "interval must be a string."
            )

        normalized = interval.strip()

        if normalized not in cls.VALID_INTERVALS:
            raise MarketDataValidationError(
                "Unsupported Binance interval: "
                f"{interval!r}"
            )

        return normalized

    @staticmethod
    def _interval_to_milliseconds(
        interval: str,
    ) -> int | None:
        """
        Convert fixed-duration Binance intervals to milliseconds.

        ``1M`` intentionally returns None because calendar months have
        variable duration.
        """

        if interval.endswith("M"):
            return None

        units = {
            "s": 1_000,
            "m": 60_000,
            "h": 3_600_000,
            "d": 86_400_000,
            "w": 604_800_000,
        }

        unit = interval[-1]

        if unit not in units:
            return None

        try:
            amount = int(
                interval[:-1]
            )
        except ValueError:
            return None

        if amount <= 0:
            return None

        return amount * units[unit]


def fetch_completed_klines(
    symbol: str,
    interval: str,
    limit: int = 500,
) -> MarketDataResult:
    """
    Convenience function for retrieving completed Spot candles.
    """

    client = BinanceMarketDataClient()

    return client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )


def self_test() -> None:
    """
    Deterministic local validation.

    This function never contacts Binance.
    """

    candles = (
        Candle(
            open_time=1_000,
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=10.0,
            close_time=59_999,
            quote_volume=1_000.0,
            number_of_trades=100,
        ),
        Candle(
            open_time=61_000,
            open=103.0,
            high=106.0,
            low=102.0,
            close=105.0,
            volume=12.0,
            close_time=119_999,
            quote_volume=1_200.0,
            number_of_trades=120,
        ),
    )

    BinanceMarketDataClient._validate_candles(
        candles,
        interval="1m",
    )

    assert candles[0].is_valid_ohlc
    assert candles[1].close > candles[0].close

    print(
        "MARKET_DATA_SELF_TEST_OK"
    )


if __name__ == "__main__":
    self_test()
