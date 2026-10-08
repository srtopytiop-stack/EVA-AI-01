"""Multi-asset Binance Spot universe selection for EVA.

The selector deliberately ranks a supplied research snapshot. It does not
hard-code BTC, ETH, or any other asset, and it does not fetch live markets.

Research flow:
    supplied Spot metadata
        -> eligibility filters
        -> cross-sectional score
        -> deterministic ranking
        -> Top-N universe

The selector returns only valid RankedAsset objects. Temporary ranking
objects are never created with an invalid rank such as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .asset_metadata import (
    AssetMarketSnapshot,
    RankedAsset,
    SpotAssetMetadata,
)


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
        if (
            isinstance(self.min_quote_volume, bool)
            or not math.isfinite(self.min_quote_volume)
            or self.min_quote_volume < 0
        ):
            raise UniverseSelectionError(
                "min_quote_volume must be finite and non-negative"
            )

        if (
            isinstance(self.max_median_spread_bps, bool)
            or math.isnan(self.max_median_spread_bps)
            or self.max_median_spread_bps < 0
        ):
            raise UniverseSelectionError(
                "max_median_spread_bps must be non-negative"
            )

        if (
            isinstance(self.min_observations, bool)
            or not isinstance(self.min_observations, int)
            or self.min_observations < 1
        ):
            raise UniverseSelectionError(
                "min_observations must be an integer >= 1"
            )

        if (
            not math.isfinite(self.min_data_quality_score)
            or not 0.0 <= self.min_data_quality_score <= 1.0
        ):
            raise UniverseSelectionError(
                "min_data_quality_score must be finite and in [0, 1]"
            )

        if (
            isinstance(self.max_assets, bool)
            or not isinstance(self.max_assets, int)
            or self.max_assets < 1
        ):
            raise UniverseSelectionError(
                "max_assets must be an integer >= 1"
            )


def _norm(value: float, maximum: float) -> float:
    """Normalize a non-negative value into [0, 1]."""

    if maximum <= 0:
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            value / maximum,
        ),
    )


def _rank_score(
    snapshot: AssetMarketSnapshot,
    eligible: Sequence[AssetMarketSnapshot],
) -> float:
    """Calculate a deterministic cross-sectional research score.

    Higher scores favor:
    - greater quote volume,
    - higher data quality,
    - lower median spread.

    Realized volatility receives only a small activity weight. EVA must not
    equate higher volatility with higher asset quality.
    """

    if not eligible:
        raise UniverseSelectionError(
            "cannot calculate a ranking score without eligible assets"
        )

    max_volume = max(
        item.quote_volume
        for item in eligible
    )

    max_volatility = max(
        item.realized_volatility
        for item in eligible
    )

    max_spread = max(
        item.median_spread_bps
        for item in eligible
    )

    liquidity = _norm(
        snapshot.quote_volume,
        max_volume,
    )

    quality = snapshot.data_quality_score

    spread_score = 1.0 - _norm(
        snapshot.median_spread_bps,
        max_spread,
    )

    activity = _norm(
        snapshot.realized_volatility,
        max_volatility,
    )

    score = (
        0.45 * liquidity
        + 0.35 * quality
        + 0.15 * spread_score
        + 0.05 * activity
    )

    return score


def select_universe(
    metadata: Iterable[SpotAssetMetadata],
    snapshots: Iterable[AssetMarketSnapshot],
    criteria: UniverseCriteria | None = None,
) -> tuple[RankedAsset, ...]:
    """Return a deterministic ranked Binance Spot research universe.

    Metadata and market snapshots are joined by symbol.

    Eligibility requires:
    - Binance exchange,
    - Spot market,
    - TRADING status,
    - sufficient quote volume,
    - acceptable median spread,
    - enough observations,
    - sufficient data quality.

    The selector is completely asset-agnostic. It does not contain a fixed
    BTC/ETH/SOL whitelist.
    """

    criteria = (
        criteria
        if criteria is not None
        else UniverseCriteria()
    )

    # ------------------------------------------------------------------
    # 1. Build the currently tradable Binance Spot metadata universe.
    # ------------------------------------------------------------------
    metadata_by_symbol: dict[str, SpotAssetMetadata] = {}

    for item in metadata:
        if (
            item.exchange != "BINANCE"
            or item.market_type != "SPOT"
        ):
            continue

        if item.is_trading:
            metadata_by_symbol[item.symbol] = item

    # ------------------------------------------------------------------
    # 2. Materialize snapshots and reject duplicate observations for
    #    the same symbol. Silent replacement would make the research
    #    universe non-deterministic from the caller's perspective.
    # ------------------------------------------------------------------
    unique_snapshots: dict[str, AssetMarketSnapshot] = {}

    for snapshot in snapshots:
        if snapshot.symbol in unique_snapshots:
            raise UniverseSelectionError(
                f"duplicate snapshot for {snapshot.symbol}"
            )

        unique_snapshots[snapshot.symbol] = snapshot

    # ------------------------------------------------------------------
    # 3. Apply objective eligibility constraints.
    # ------------------------------------------------------------------
    eligible = [
        snapshot
        for snapshot in unique_snapshots.values()
        if (
            snapshot.symbol in metadata_by_symbol
            and snapshot.quote_volume
            >= criteria.min_quote_volume
            and snapshot.median_spread_bps
            <= criteria.max_median_spread_bps
            and snapshot.observation_count
            >= criteria.min_observations
            and snapshot.data_quality_score
            >= criteria.min_data_quality_score
        )
    ]

    if not eligible:
        return ()

    # ------------------------------------------------------------------
    # 4. Calculate scores BEFORE constructing RankedAsset.
    #
    #    This is the critical fix:
    #    RankedAsset requires rank >= 1, so we never create an invalid
    #    temporary object with rank=0 just to sort it.
    # ------------------------------------------------------------------
    scored: list[
        tuple[AssetMarketSnapshot, float]
    ] = [
        (
            item,
            _rank_score(
                item,
                eligible,
            ),
        )
        for item in eligible
    ]

    # Deterministic ordering:
    #   1. highest score first
    #   2. symbol ascending as a stable tie-breaker
    scored.sort(
        key=lambda pair: (
            -pair[1],
            pair[0].symbol,
        )
    )

    # ------------------------------------------------------------------
    # 5. Limit the research universe to Top-N.
    # ------------------------------------------------------------------
    selected = scored[
        : criteria.max_assets
    ]

    # ------------------------------------------------------------------
    # 6. Create only valid RankedAsset objects with final ranks 1..N.
    # ------------------------------------------------------------------
    return tuple(
        RankedAsset(
            symbol=item.symbol,
            score=score,
            rank=rank,
        )
        for rank, (item, score)
        in enumerate(
            selected,
            start=1,
        )
    )
