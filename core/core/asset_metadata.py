"""Asset metadata contracts for multi-asset Spot research."""

from __future__ import annotations

from dataclasses import dataclass
import math


class AssetMetadataError(ValueError):
    """Raised when asset metadata is invalid."""


def _non_empty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssetMetadataError(f"{field} must be a non-empty string")
    return value.strip().upper()


def _finite_non_negative(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AssetMetadataError(f"{field} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class SpotAssetMetadata:
    """Static or slowly changing metadata for a Binance Spot symbol."""

    symbol: str
    base_asset: str
    quote_asset: str
    exchange: str = "BINANCE"
    market_type: str = "SPOT"
    status: str = "TRADING"
    price_step: float | None = None
    quantity_step: float | None = None
    min_notional: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _non_empty(self.symbol, "symbol"))
        object.__setattr__(self, "base_asset", _non_empty(self.base_asset, "base_asset"))
        object.__setattr__(self, "quote_asset", _non_empty(self.quote_asset, "quote_asset"))
        object.__setattr__(self, "exchange", _non_empty(self.exchange, "exchange"))
        object.__setattr__(self, "market_type", _non_empty(self.market_type, "market_type"))
        object.__setattr__(self, "status", _non_empty(self.status, "status"))

        if self.market_type != "SPOT":
            raise AssetMetadataError("EVA research universe is Spot-only")
        if self.exchange != "BINANCE":
            raise AssetMetadataError("this contract currently targets Binance Spot")
        if self.base_asset == self.quote_asset:
            raise AssetMetadataError("base_asset and quote_asset must differ")

        for field in ("price_step", "quantity_step", "min_notional"):
            value = getattr(self, field)
            if value is not None:
                numeric = _finite_non_negative(value, field)
                if numeric == 0:
                    raise AssetMetadataError(f"{field} must be greater than zero when provided")
                object.__setattr__(self, field, numeric)

    @property
    def is_trading(self) -> bool:
        return self.status == "TRADING"


@dataclass(frozen=True, slots=True)
class AssetMarketSnapshot:
    """Research-time market statistics used for universe eligibility/ranking."""

    symbol: str
    quote_volume: float
    median_spread_bps: float
    realized_volatility: float
    observation_count: int
    data_quality_score: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _non_empty(self.symbol, "symbol"))
        for field in ("quote_volume", "median_spread_bps", "realized_volatility", "data_quality_score"):
            numeric = _finite_non_negative(getattr(self, field), field)
            object.__setattr__(self, field, numeric)
        if self.observation_count < 1:
            raise AssetMetadataError("observation_count must be at least 1")
        if self.data_quality_score > 1:
            raise AssetMetadataError("data_quality_score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RankedAsset:
    """Output of the universe selector."""

    symbol: str
    score: float
    rank: int
