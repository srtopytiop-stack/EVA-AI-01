"""
EVA-AI-01
=========
Market Data Layer - Phase 3

Responsibilities:
- Retrieve public Binance Spot OHLCV data.
- Normalize exchange responses into deterministic records.
- Validate chronological integrity.
- Detect duplicates and missing intervals.
- Identify the last completed candle.
- Prevent accidental use of an incomplete candle.
- Keep the data layer independent from trading execution.

Design principles:
- No API keys are required for public market data.
- No trading orders are placed here.
- No signal generation is performed here.
- No future information is introduced.
- All timestamps are UTC milliseconds internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


BINANCE_SPOT_BASE_URL = "https://api.binance.com"

KLINES_ENDPOINT = "/api/v3/klines"


class MarketDataError(RuntimeError):
    """Base exception for market-data failures."""


class MarketDataRequestError(MarketDataError):
    """Raised when Binance cannot be queried successfully."""


class MarketDataValidationError(MarketDataError):
    """Raised when returned market data fails validation."""


@dataclass(frozen=True)
class Candle:
    """
    Immutable OHLCV candle.

    All timestamps are UTC milliseconds.
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
        """Return candle open time as a UTC datetime."""
        return datetime.fromtimestamp(
            self.open_time / 1000,
            tz=timezone.utc,
        )

    @property
    def is_valid_ohlc(self) -> bool:
        """Validate basic OHLC relationships."""

        return (
            self.low <= self.open <= self.high
            and self.low <= self.close <= self.high
            and self.low > 0
            and self.high > 0
            and self.open > 0
            and self.close > 0
        )


@dataclass(frozen=True)
class MarketDataResult:
    """
    Validated market-data response.

    The result contains only completed candles unless explicitly
    requested otherwise.
    """

    symbol: str
    interval: str
    candles: tuple[Candle, ...]

    @property
    def count(self) -> int:
        return len(self.candles)

    @property
    def last_candle(self) -> Candle | None:
        if not self.candles:
            return None

        return self.candles[-1]


class BinanceMarketDataClient:
    """
    Public Binance Spot market-data client.

    This client intentionally does not contain:
    - API keys
    - API secrets
    - order execution
    - portfolio logic
    - trading decisions
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
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
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

        Parameters
        ----------
        symbol:
            Binance symbol, e.g. BTCUSDT.

        interval:
            Binance candle interval.

        limit:
            Number of candles requested.

        end_time:
            Optional UTC timestamp in milliseconds.
        """

        normalized_symbol = self._validate_symbol(symbol)
        normalized_interval = self._validate_interval(interval)

        if not 1 <= limit <= 1000:
            raise MarketDataValidationError(
                "limit must be between 1 and 1000."
            )

        params: dict[str, Any] = {
            "symbol": normalized_symbol,
            "interval": normalized_interval,
            "limit": limit,
        }

        if end_time is not None:
            if end_time <= 0:
                raise MarketDataValidationError(
                    "end_time must be a positive Unix timestamp "
                    "in milliseconds."
                )

            params["endTime"] = int(end_time)

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
        """Perform the public Binance request."""

        url = f"{self.base_url}{KLINES_ENDPOINT}"

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
                f"{response.status_code}: {response.text[:300]}"
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
    def _parse_candle(row: list[Any]) -> Candle:
        """Convert one Binance kline row into a Candle."""

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

        except (TypeError, ValueError) as exc:
            raise MarketDataValidationError(
                "Malformed Binance kline: numeric conversion failed."
            ) from exc

    @classmethod
    def _validate_candles(
        cls,
        candles: tuple[Candle, ...],
        interval: str,
    ) -> None:
        """Validate structural integrity of candle data."""

        if not candles:
            raise MarketDataValidationError(
                "Binance returned no candles."
            )

        previous_time: int | None = None

        for candle in candles:
            if not candle.is_valid_ohlc:
                raise MarketDataValidationError(
                    f"Invalid OHLC relationship at "
                    f"{candle.open_time}."
                )

            if candle.volume < 0:
                raise MarketDataValidationError(
                    f"Negative volume at {candle.open_time}."
                )

            if candle.quote_volume < 0:
                raise MarketDataValidationError(
                    f"Negative quote volume at {candle.open_time}."
                )

            if candle.number_of_trades < 0:
                raise MarketDataValidationError(
                    f"Negative trade count at {candle.open_time}."
                )

            if candle.close_time < candle.open_time:
                raise MarketDataValidationError(
                    f"Invalid candle timestamps at "
                    f"{candle.open_time}."
                )

            if previous_time is not None:
                if candle.open_time <= previous_time:
                    raise MarketDataValidationError(
                        "Candle timestamps are not strictly increasing."
                    )

            previous_time = candle.open_time

        # Verify interval continuity where Binance provides
        # regular interval spacing.
        interval_ms = cls._interval_to_milliseconds(interval)

        if interval_ms is not None:
            for previous, current in zip(
                candles,
                candles[1:],
            ):
                difference = current.open_time - previous.open_time

                if difference != interval_ms:
                    raise MarketDataValidationError(
                        "Missing or irregular candle interval detected: "
                        f"{previous.open_time} -> {current.open_time}."
                    )

    @staticmethod
    def _is_completed(candle: Candle) -> bool:
        """
        Determine whether a candle has fully closed.

        Binance's latest returned candle may still be forming.
        Using it for signals can introduce unstable observations.
        """

        now_ms = int(
            datetime.now(timezone.utc).timestamp() * 1000
        )

        return candle.close_time < now_ms

    @classmethod
    def _validate_symbol(cls, symbol: str) -> str:
        """Validate and normalize a trading symbol."""

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
    def _validate_interval(cls, interval: str) -> str:
        """Validate Binance interval."""

        if not isinstance(interval, str):
            raise MarketDataValidationError(
                "interval must be a string."
            )

        normalized = interval.strip()

        if normalized not in cls.VALID_INTERVALS:
            raise MarketDataValidationError(
                f"Unsupported Binance interval: {interval!r}"
            )

        return normalized

    @staticmethod
    def _interval_to_milliseconds(
        interval: str,
    ) -> int | None:
        """Convert Binance interval to milliseconds."""

        units = {
            "s": 1_000,
            "m": 60_000,
            "h": 3_600_000,
            "d": 86_400_000,
            "w": 604_800_000,
        }

        if interval.endswith("M"):
            # Calendar-month candles do not have a fixed duration.
            return None

        unit = interval[-1]

        if unit not in units:
            return None

        try:
            amount = int(interval[:-1])
        except ValueError:
            return None

        return amount * units[unit]


def fetch_completed_klines(
    symbol: str,
    interval: str,
    limit: int = 500,
) -> MarketDataResult:
    """
    Convenience function for retrieving completed candles.
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

    This test does not contact Binance.
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

    print("MARKET_DATA_SELF_TEST_OK")


if __name__ == "__main__":
    self_test()
