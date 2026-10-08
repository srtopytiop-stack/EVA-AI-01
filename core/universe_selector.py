"""Multi-asset Binance Spot universe selection for EVA.

The selector deliberately ranks a supplied research snapshot. It does not
hard-code BTC, ETH, or any other asset, and it does not fetch live markets.
Live discovery can be added later behind this contract without changing the
research semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .asset_metadata import AssetMarketSnapshot, RankedAsset, SpotAssetMetadata


class UniverseSelectionError(ValueError):
    """Raised when universe selection inputs are invalid."""


@dataclass(frozen=True, slots=True)
class UniverseCriteria:
    """Eligibility and ranking parameters for Spot research assets."""

    min_quote_volume: float = 0.0
    max_median_spread_bps: float = math.inf
    min_observations: int = 1
    min_data_quality_score: float = 0.0
    max_assets: int = 20

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_quote_volume) or self.min_quote_volume < 0:
            raise UniverseSelectionError("min_quote_volume must be finite and non-negative")
        if self.max_median_spread_bps < 0:
            raise UniverseSelectionError("max_median_spread_bps must be non-negative")
        if self.min_observations < 1:
            raise UniverseSelectionError("min_observations must be at least 1")
        if not 0 <= self.min_data_quality_score <= 1:
            raise UniverseSelectionError("min_data_quality_score must be in [0, 1]")
        if self.max_assets < 1:
            raise UniverseSelectionError("max_assets must be at least 1")


def _norm(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, value / maximum))


def _rank_score(snapshot: AssetMarketSnapshot, eligible: Sequence[AssetMarketSnapshot]) -> float:
    """Rank higher liquidity/quality and lower spread, using cross-section normalization."""
    max_volume = max(item.quote_volume for item in eligible)
    max_volatility = max(item.realized_volatility for item in eligible)
    max_spread = max(item.median_spread_bps for item in eligible)

    liquidity = _norm(snapshot.quote_volume, max_volume)
    quality = snapshot.data_quality_score
    spread_score = 1.0 - _norm(snapshot.median_spread_bps, max_spread)

    # Volatility receives only a modest weight: EVA should not equate
    # "more volatile" with "better". It is present only to avoid ignoring
    # assets with effectively zero movement when all else is comparable.
    activity = _norm(snapshot.realized_volatility, max_volatility)

    return 0.45 * liquidity + 0.35 * quality + 0.15 * spread_score + 0.05 * activity


def select_universe(
    metadata: Iterable[SpotAssetMetadata],
    snapshots: Iterable[AssetMarketSnapshot],
    criteria: UniverseCriteria | None = None,
) -> tuple[RankedAsset, ...]:
    """Return a deterministic ranked Spot research universe.

    Metadata and snapshots are joined by symbol. Only Binance Spot symbols
    with status ``TRADING`` and criteria-compliant market snapshots qualify.
    """
    criteria = criteria or UniverseCriteria()

    metadata_by_symbol: dict[str, SpotAssetMetadata] = {}
    for item in metadata:
        if item.exchange != "BINANCE" or item.market_type != "SPOT":
            continue
        if item.is_trading:
            metadata_by_symbol[item.symbol] = item

    unique_snapshots: dict[str, AssetMarketSnapshot] = {}
    for snapshot in snapshots:
        if snapshot.symbol in unique_snapshots:
            raise UniverseSelectionError(f"duplicate snapshot for {snapshot.symbol}")
        unique_snapshots[snapshot.symbol] = snapshot

    eligible = [
        snapshot
        for snapshot in unique_snapshots.values()
        if snapshot.symbol in metadata_by_symbol
        and snapshot.quote_volume >= criteria.min_quote_volume
        and snapshot.median_spread_bps <= criteria.max_median_spread_bps
        and snapshot.observation_count >= criteria.min_observations
        and snapshot.data_quality_score >= criteria.min_data_quality_score
    ]

    ranked = sorted(
        (
            RankedAsset(symbol=item.symbol, score=_rank_score(item, eligible), rank=0)
            for item in eligible
        ),
        key=lambda item: (-item.score, item.symbol),
    )

    selected = ranked[: criteria.max_assets]
    return tuple(
        RankedAsset(symbol=item.symbol, score=item.score, rank=index)
        for index, item in enumerate(selected, start=1)
    )
