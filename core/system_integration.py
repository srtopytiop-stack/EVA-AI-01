"""
EVA-AI-01 - System Integration Layer
====================================
Phase 14-A / Phase 15-B research integration boundary.

The integration layer composes the research components while preserving
their individual contracts.

A flat equity curve is a valid research result. When statistical inference
is undefined, the pipeline preserves the backtest and records the failure
instead of manufacturing a statistic.

This module is research-only. It does not place live orders, use Binance
credentials, or send Telegram messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import pandas as pd

from core.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from core.backtest_validation import validate_backtest_result
from core.bootstrap_validation import (
    MIN_BOOTSTRAP_REPLICATIONS,
    BootstrapValidationError,
    BootstrapValidationReport,
    validate_backtest_result_with_bootstrap,
)
from core.statistical_validation import (
    StatisticalValidationError,
    StatisticalValidationReport,
)
from data.market_data import Candle, MarketDataResult
from features.feature_engine import (
    FeatureConfig,
    build_features,
    feature_columns,
    validate_feature_output,
)


class IntegrationError(ValueError):
    """Raised when the system-level integration boundary is invalid."""


@dataclass(frozen=True)
class IntegrationConfig:
    """Immutable configuration for one integrated research run."""

    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    number_of_trials: int = 2
    benchmark_sharpe: float = 0.0
    bootstrap_replications: int = MIN_BOOTSTRAP_REPLICATIONS
    bootstrap_block_length: int = 5
    bootstrap_seed: int | None = 42
    risk_free_per_period: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.number_of_trials, bool)
            or not isinstance(self.number_of_trials, int)
            or self.number_of_trials < 2
        ):
            raise IntegrationError(
                "number_of_trials must be an integer >= 2"
            )

        _finite_number(self.benchmark_sharpe, "benchmark_sharpe")

        risk_free = _finite_number(
            self.risk_free_per_period,
            "risk_free_per_period",
        )
        if risk_free <= -1.0:
            raise IntegrationError(
                "risk_free_per_period must be > -1"
            )

        if (
            isinstance(self.bootstrap_replications, bool)
            or not isinstance(self.bootstrap_replications, int)
            or self.bootstrap_replications < MIN_BOOTSTRAP_REPLICATIONS
        ):
            raise IntegrationError(
                "bootstrap_replications must be >= "
                f"{MIN_BOOTSTRAP_REPLICATIONS}"
            )

        if (
            isinstance(self.bootstrap_block_length, bool)
            or not isinstance(self.bootstrap_block_length, int)
            or self.bootstrap_block_length < 2
        ):
            raise IntegrationError(
                "bootstrap_block_length must be an integer >= 2"
            )

        if self.bootstrap_seed is not None and (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
        ):
            raise IntegrationError(
                "bootstrap_seed must be an integer or None"
            )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise IntegrationError(f"{name} must be numeric")

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationError(f"{name} must be numeric") from exc

    if not isfinite(normalized):
        raise IntegrationError(f"{name} must be finite")

    return normalized


@dataclass(frozen=True)
class IntegrationResult:
    """Immutable summary of one integrated research run.

    Statistical and bootstrap reports may be absent when inference is
    mathematically undefined. Missing evidence is represented explicitly;
    no statistic is fabricated.
    """

    symbol: str
    interval: str
    candle_count: int
    feature_rows: int
    feature_columns: tuple[str, ...]
    backtest: BacktestResult
    statistical_validation: StatisticalValidationReport | None
    bootstrap_validation: BootstrapValidationReport | None
    statistical_validation_error: str | None = None
    bootstrap_validation_error: str | None = None

    @property
    def final_equity(self) -> float:
        return float(self.backtest.metrics.final_equity)

    @property
    def validation_complete(self) -> bool:
        """Whether both statistical inference layers produced reports."""
        return (
            self.statistical_validation is not None
            and self.bootstrap_validation is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "candle_count": self.candle_count,
            "feature_rows": self.feature_rows,
            "feature_columns": list(self.feature_columns),
            "backtest": self.backtest.to_dict(),
            "statistical_validation": (
                self.statistical_validation.to_dict()
                if self.statistical_validation is not None
                else None
            ),
            "bootstrap_validation": (
                self.bootstrap_validation.to_dict()
                if self.bootstrap_validation is not None
                else None
            ),
            "statistical_validation_error": (
                self.statistical_validation_error
            ),
            "bootstrap_validation_error": (
                self.bootstrap_validation_error
            ),
            "validation_complete": self.validation_complete,
            "final_equity": self.final_equity,
        }


def _candles_to_frame(
    candles: tuple[Candle, ...],
) -> pd.DataFrame:
    """Convert validated candles into FeatureEngine's OHLCV contract.

    Candle timestamps use Unix milliseconds. The unit and timezone must be
    explicit; otherwise an integer timestamp may be interpreted as
    nanoseconds and silently produce an incorrect date near 1970.

    The resulting timestamp represents the candle's opening time. It does
    not imply that the completed candle's final OHLCV values were available
    at that opening time.
    """

    timestamps = pd.to_datetime(
        [candle.open_time for candle in candles],
        unit="ms",
        utc=True,
        errors="raise",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(candle.open) for candle in candles],
            "high": [float(candle.high) for candle in candles],
            "low": [float(candle.low) for candle in candles],
            "close": [float(candle.close) for candle in candles],
            "volume": [float(candle.volume) for candle in candles],
        },
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )


def _validate_market_data_result(
    market_data: MarketDataResult,
) -> tuple[str, str, tuple[Candle, ...]]:
    """Validate the structural contract at the integration boundary."""

    if not isinstance(market_data, MarketDataResult):
        raise IntegrationError(
            "market_data must be a MarketDataResult"
        )

    symbol = market_data.symbol.strip().upper()
    interval = market_data.interval.strip()
    candles = tuple(market_data.candles)

    if not symbol:
        raise IntegrationError(
            "market_data.symbol must be non-empty"
        )

    if not interval:
        raise IntegrationError(
            "market_data.interval must be non-empty"
        )

    if len(candles) < 2:
        raise IntegrationError(
            "market_data must contain at least two candles"
        )

    if any(not isinstance(candle, Candle) for candle in candles):
        raise IntegrationError(
            "market_data.candles must contain only Candle objects"
        )

    previous_open: int | None = None

    for index, candle in enumerate(candles):
        if not candle.is_valid_ohlc:
            raise IntegrationError(
                f"invalid OHLC candle at index {index}"
            )

        if candle.close_time <= candle.open_time:
            raise IntegrationError(
                f"invalid candle timestamps at index {index}"
            )

        if (
            previous_open is not None
            and candle.open_time <= previous_open
        ):
            raise IntegrationError(
                "market_data.candles must be strictly chronological"
            )

        previous_open = candle.open_time

    return symbol, interval, candles


def run_integrated_research(
    market_data: MarketDataResult,
    *,
    config: IntegrationConfig | None = None,
) -> IntegrationResult:
    """Run the offline integration pipeline for one Spot market-data result.

    The feature layer and the event-driven backtest both receive information
    derived from the same completed-candle dataset.

    Statistical inference fails softly only for the documented domain
    errors. A failure is preserved in the result for downstream safeguards.
    """

    cfg = config or IntegrationConfig()

    symbol, interval, candles = _validate_market_data_result(
        market_data
    )

    # 1. Feature engineering on the supplied completed candles.
    frame = _candles_to_frame(candles)

    try:
        generated = build_features(
            frame,
            cfg.feature_config,
        )
        validate_feature_output(generated)
    except Exception as exc:
        raise IntegrationError(
            f"feature-engine integration failed: {exc}"
        ) from exc

    if len(generated) != len(candles):
        raise IntegrationError(
            "feature output row count does not match candle count"
        )

    columns = tuple(feature_columns(generated))

    # 2. Existing event-driven backtest remains authoritative.
    # The feature DataFrame is validated and reported but is not yet
    # consumed by BacktestEngine as an input.
    try:
        backtest = BacktestEngine(
            cfg.backtest_config
        ).run(
            candles,
            symbol=symbol,
        )
    except Exception as exc:
        raise IntegrationError(
            f"backtest integration failed: {exc}"
        ) from exc

    # 3. Statistical validation after the backtest has completed.
    statistical: StatisticalValidationReport | None = None
    statistical_error: str | None = None

    try:
        statistical = validate_backtest_result(
            backtest,
            number_of_trials=cfg.number_of_trials,
            benchmark_sharpe=cfg.benchmark_sharpe,
        )
    except StatisticalValidationError as exc:
        statistical_error = str(exc)

    # 4. Dependence-aware bootstrap on the completed equity curve.
    bootstrap: BootstrapValidationReport | None = None
    bootstrap_error: str | None = None

    try:
        bootstrap = validate_backtest_result_with_bootstrap(
            backtest,
            bootstrap_replications=cfg.bootstrap_replications,
            block_length=cfg.bootstrap_block_length,
            seed=cfg.bootstrap_seed,
            risk_free_per_period=cfg.risk_free_per_period,
        )
    except BootstrapValidationError as exc:
        bootstrap_error = str(exc)

    return IntegrationResult(
        symbol=symbol,
        interval=interval,
        candle_count=len(candles),
        feature_rows=len(generated),
        feature_columns=columns,
        backtest=backtest,
        statistical_validation=statistical,
        bootstrap_validation=bootstrap,
        statistical_validation_error=statistical_error,
        bootstrap_validation_error=bootstrap_error,
    )


__all__ = [
    "IntegrationConfig",
    "IntegrationError",
    "IntegrationResult",
    "run_integrated_research",
]
