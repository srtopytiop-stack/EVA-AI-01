
"""Dynamic Binance Spot universe data acquisition for EVA.

Uses public market-data endpoints only.
Never accesses accounts or submits orders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import requests

from core.asset_metadata import (
    AssetMarketSnapshot,
    SpotAssetMetadata,
)
from data.market_data import (
    BINANCE_SPOT_BASE_URL,
    BinanceMarketDataClient,
    MarketDataError,
)


class SpotUniverseDataError(RuntimeError):
    """Raised when public universe data cannot be validated."""


@dataclass(frozen=True)
class SpotUniverseData:
    """Metadata and market snapshots ready for universe selection."""

    metadata: tuple[SpotAssetMetadata, ...]
    snapshots: tuple[AssetMarketSnapshot, ...]
    candidates_examined: int

    def __post_init__(self) -> None:
        metadata_symbols = [item.symbol for item in self.metadata]
        snapshot_symbols = [item.symbol for item in self.snapshots]

        if len(metadata_symbols) != len(set(metadata_symbols)):
            raise SpotUniverseDataError(
                "duplicate symbols in universe metadata"
            )

        if len(snapshot_symbols) != len(set(snapshot_symbols)):
            raise SpotUniverseDataError(
                "duplicate symbols in market snapshots"
            )


class BinanceSpotUniverseClient:
    """Discover and measure Binance Spot USDT pairs using public endpoints."""

    def __init__(
        self,
        *,
        base_url: str = BINANCE_SPOT_BASE_URL,
        timeout: float = 10.0,
        session: requests.Session | None = None,
        market_data_client: BinanceMarketDataClient | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise SpotUniverseDataError(
                "base_url must be a non-empty string"
            )

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise SpotUniverseDataError(
                "timeout must be finite and greater than zero"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        self.market_data_client = (
            market_data_client
            or BinanceMarketDataClient(
                base_url=self.base_url,
                timeout=self.timeout,
                session=self.session,
            )
        )

    def _get_json(self, path: str) -> Any:
        """Fetch one public JSON endpoint with explicit error handling."""

        try:
            response = self.session.get(
                f"{self.base_url}{path}",
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise SpotUniverseDataError(
                f"public market-data request failed for {path}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise SpotUniverseDataError(
                f"Binance returned HTTP {response.status_code} "
                f"for {path}: {response.text[:200]}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise SpotUniverseDataError(
                f"Binance returned invalid JSON for {path}"
            ) from exc

    @staticmethod
    def _positive_number(value: Any, field: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SpotUniverseDataError(
                f"{field} must be numeric"
            ) from exc

        if not math.isfinite(number) or number <= 0:
            raise SpotUniverseDataError(
                f"{field} must be finite and positive"
            )

        return number

    def discover(
        self,
        *,
        interval: str = "1h",
        candidate_limit: int = 40,
        candles_per_symbol: int = 100,
        quote_asset: str = "USDT",
    ) -> SpotUniverseData:
        """Discover liquid Spot pairs and build measured market snapshots.

        Candidate discovery is market-wide, not a fixed coin whitelist.
        Historical candles are requested only for the most liquid candidates
        to keep request volume bounded.
        """

        if (
            isinstance(candidate_limit, bool)
            or not isinstance(candidate_limit, int)
            or not 1 <= candidate_limit <= 100
        ):
            raise SpotUniverseDataError(
                "candidate_limit must be an integer from 1 to 100"
            )

        if (
            isinstance(candles_per_symbol, bool)
            or not isinstance(candles_per_symbol, int)
            or not 30 <= candles_per_symbol <= 1000
        ):
            raise SpotUniverseDataError(
                "candles_per_symbol must be an integer from 30 to 1000"
            )

        if not isinstance(quote_asset, str) or not quote_asset.strip().isalnum():
            raise SpotUniverseDataError(
                "quote_asset must be a non-empty alphanumeric string"
            )

        quote_asset = quote_asset.strip().upper()

        exchange_info = self._get_json("/api/v3/exchangeInfo")
        tickers = self._get_json("/api/v3/ticker/24hr")
        book_tickers = self._get_json("/api/v3/ticker/bookTicker")

        if not isinstance(exchange_info, dict):
            raise SpotUniverseDataError(
                "exchangeInfo must be a JSON object"
            )

        symbols_payload = exchange_info.get("symbols")
        if not isinstance(symbols_payload, list):
            raise SpotUniverseDataError(
                "exchangeInfo is missing its symbols list"
            )

        if not isinstance(tickers, list) or not isinstance(book_tickers, list):
            raise SpotUniverseDataError(
                "ticker endpoints must return JSON lists"
            )

        metadata_by_symbol: dict[str, SpotAssetMetadata] = {}

        for item in symbols_payload:
            if not isinstance(item, dict):
                continue

            symbol = item.get("symbol")
            base = item.get("baseAsset")
            quote = item.get("quoteAsset")

            if (
                not isinstance(symbol, str)
                or not isinstance(base, str)
                or not isinstance(quote, str)
            ):
                continue

            if (
                item.get("status") != "TRADING"
                or item.get("isSpotTradingAllowed") is False
                or quote.upper() != quote_asset
                or base.upper() == quote_asset
            ):
                continue

            try:
                metadata_by_symbol[symbol.upper()] = SpotAssetMetadata(
                    symbol=symbol,
                    base_asset=base,
                    quote_asset=quote,
                    exchange="BINANCE",
                    market_type="SPOT",
                    status="TRADING",
                )
            except ValueError:
                # Invalid exchange metadata is not allowed into the universe.
                continue

        ticker_by_symbol: dict[str, dict[str, Any]] = {}

        for item in tickers:
            if not isinstance(item, dict):
                continue

            symbol = item.get("symbol")
            if isinstance(symbol, str) and symbol in metadata_by_symbol:
                ticker_by_symbol[symbol] = item

        spread_by_symbol: dict[str, float] = {}

        for item in book_tickers:
            if not isinstance(item, dict):
                continue

            symbol = item.get("symbol")
            if not isinstance(symbol, str) or symbol not in metadata_by_symbol:
                continue

            try:
                bid = self._positive_number(item.get("bidPrice"), "bidPrice")
                ask = self._positive_number(item.get("askPrice"), "askPrice")
            except SpotUniverseDataError:
                continue

            if ask < bid:
                continue

            midpoint = (bid + ask) / 2.0
            spread_by_symbol[symbol] = (
                (ask - bid) / midpoint
            ) * 10_000.0

        # Keep only pairs with complete ticker and bid/ask information.
        candidates: list[tuple[str, float, float]] = []

        for symbol, item in ticker_by_symbol.items():
            spread = spread_by_symbol.get(symbol)
            if spread is None:
                continue

            try:
                quote_volume = float(item.get("quoteVolume"))
            except (TypeError, ValueError, OverflowError):
                continue

            if not math.isfinite(quote_volume) or quote_volume <= 0:
                continue

            candidates.append((symbol, quote_volume, spread))

        # A stable, market-wide preselection limits historical requests.
        candidates.sort(key=lambda row: (-row[1], row[0]))
        candidates = candidates[:candidate_limit]

        snapshots: list[AssetMarketSnapshot] = []

        for symbol, quote_volume, spread_bps in candidates:
            try:
                market = self.market_data_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=candles_per_symbol,
                )
            except (MarketDataError, ValueError):
                # A failed pair is excluded rather than assigned fake data.
                continue

            candles = market.candles

            if len(candles) < 30:
                continue

            closes = [candle.close for candle in candles]

            if any(
                not math.isfinite(price) or price <= 0
                for price in closes
            ):
                continue

            log_returns = [
                math.log(current / previous)
                for previous, current in zip(closes, closes[1:])
            ]

            if len(log_returns) < 29:
                continue

            mean_return = sum(log_returns) / len(log_returns)
            variance = sum(
                (value - mean_return) ** 2
                for value in log_returns
            ) / (len(log_returns) - 1)

            volatility = math.sqrt(max(0.0, variance))

            snapshots.append(
                AssetMarketSnapshot(
                    symbol=symbol,
                    quote_volume=quote_volume,
                    median_spread_bps=spread_bps,
                    realized_volatility=volatility,
                    observation_count=len(log_returns),
                    data_quality_score=1.0,
                )
            )

        return SpotUniverseData(
            metadata=tuple(
                metadata_by_symbol[symbol]
                for symbol, _, _ in candidates
                if symbol in metadata_by_symbol
            ),
            snapshots=tuple(snapshots),
            candidates_examined=len(candidates),
        )


__all__ = [
    "BinanceSpotUniverseClient",
    "SpotUniverseData",
    "SpotUniverseDataError",
]
