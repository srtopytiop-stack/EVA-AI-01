"""
EVA-AI-01 - System Integration Layer
====================================
Phase 14-A preparation.

Orchestrates the completed research components without modifying the stable
engines:

MarketDataResult -> FeatureEngine -> BacktestEngine -> Statistical Validation
                  -> Moving-Block Bootstrap Validation

Feature output is retained as an auditable research artifact. It is not
silently injected into SignalEngine because SignalEngine's current public
contract consumes OHLCV history directly. Changing that contract is a
separate architectural decision.

This module is offline/research-only. It does not place live orders, use
Binance credentials, or send Telegram messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import pandas as pd

from core.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from core.backtest_validation import validate_backtest_result
from core.bootstrap_validation import (
    MIN_BOOTSTRAP_REPLICATIONS,
    BootstrapValidationReport,
    validate_backtest_result_with_bootstrap,
)
from core.statistical_validation import StatisticalValidationReport
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
            raise IntegrationError("number_of_trials must be an integer >= 2")

        benchmark = _finite_number(self.benchmark_sharpe, "benchmark_sharpe")
        risk_free = _finite_number(
            self.risk_free_per_period,
            "risk_free_per_period",
        )
        if risk_free <= -1.0:
            raise IntegrationError("risk_free_per_period must be > -1")

        if (
            isinstance(self.bootstrap_replications, bool)
            or not isinstance(self.bootstrap_replications, int)
            or self.bootstrap_replications < MIN_BOOTSTRAP_REPLICATIONS
        ):
            raise IntegrationError(
                f"bootstrap_replications must be >= {MIN_BOOTSTRAP_REPLICATIONS}"
            )

        if (
            isinstance(self.bootstrap_block_length, bool)
            or not isinstance(self.bootstrap_block_length, int)
            or self.bootstrap_block_length < 2
        ):
            raise IntegrationError("bootstrap_block_length must be an integer >= 2")

        if self.bootstrap_seed is not None and (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
        ):
            raise IntegrationError("bootstrap_seed must be an integer or None")

        # Keep explicit local bindings so validation is not accidentally
        # optimized away and to document the boundary values.
        _ = benchmark


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
    """Immutable summary of one complete integrated research run."""

    symbol: str
    interval: str
    candle_count: int
    feature_rows: int
    feature_columns: tuple[str, ...]
    backtest: BacktestResult
    statistical_validation: StatisticalValidationReport
    bootstrap_validation: BootstrapValidationReport

    @property
    def final_equity(self) -> float:
        return float(self.backtest.metrics.final_equity)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "candle_count": self.candle_count,
            "feature_rows": self.feature_rows,
            "feature_columns": list(self.feature_columns),
            "backtest": self.backtest.to_dict(),
            "statistical_validation": self.statistical_validation.to_dict(),
            "bootstrap_validation": self.bootstrap_validation.to_dict(),
            "final_equity": self.final_equity,
        }


def _candles_to_frame(candles: tuple[Candle, ...]) -> pd.DataFrame:
    """Convert validated candles to FeatureEngine's OHLCV contract."""
    return pd.DataFrame(
        [
            {
                "timestamp": candle.open_time,
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
            }
            for candle in candles
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


def _validate_market_data_result(
    market_data: MarketDataResult,
) -> tuple[str, str, tuple[Candle, ...]]:
    if not isinstance(market_data, MarketDataResult):
        raise IntegrationError("market_data must be a MarketDataResult")

    symbol = market_data.symbol.strip().upper()
    interval = market_data.interval.strip()
    candles = tuple(market_data.candles)

    if not symbol:
        raise IntegrationError("market_data.symbol must be non-empty")
    if not interval:
        raise IntegrationError("market_data.interval must be non-empty")
    if len(candles) < 2:
        raise IntegrationError("market_data must contain at least two candles")
    if any(not isinstance(candle, Candle) for candle in candles):
        raise IntegrationError("market_data.candles must contain only Candle objects")

    previous_open: int | None = None
    for index, candle in enumerate(candles):
        if not candle.is_valid_ohlc:
            raise IntegrationError(f"invalid OHLC candle at index {index}")
        if candle.close_time <= candle.open_time:
            raise IntegrationError(f"invalid candle timestamps at index {index}")
        if previous_open is not None and candle.open_time <= previous_open:
            raise IntegrationError("market_data.candles must be strictly chronological")
        previous_open = candle.open_time

    return symbol, interval, candles


def run_integrated_research(
    market_data: MarketDataResult,
    *,
    config: IntegrationConfig | None = None,
) -> IntegrationResult:
    """Run the complete offline integration pipeline."""
    cfg = config or IntegrationConfig()
    symbol, interval, candles = _validate_market_data_result(market_data)

    # 1. Feature layer: same completed candles, no future data.
    frame = _candles_to_frame(candles)
    try:
        generated = build_features(frame, cfg.feature_config)
        validate_feature_output(generated)
    except Exception as exc:
        raise IntegrationError(f"feature-engine integration failed: {exc}") from exc

    if len(generated) != len(candles):
        raise IntegrationError("feature output row count does not match candle count")

    columns = tuple(feature_columns(generated))

    # 2. Backtest layer: the existing event-driven engine remains authoritative.
    try:
        backtest = BacktestEngine(cfg.backtest_config).run(
            candles,
            symbol=symbol,
        )
    except Exception as exc:
        raise IntegrationError(f"backtest integration failed: {exc}") from exc

    # 3. Statistical validation: post-backtest only.
    try:
        statistical = validate_backtest_result(
            backtest,
            number_of_trials=cfg.number_of_trials,
            benchmark_sharpe=cfg.benchmark_sharpe,
        )
    except Exception as exc:
        raise IntegrationError(
            f"statistical-validation integration failed: {exc}"
        ) from exc

    # 4. Dependence-aware bootstrap: consumes only the completed equity curve.
    try:
        bootstrap = validate_backtest_result_with_bootstrap(
            backtest,
            bootstrap_replications=cfg.bootstrap_replications,
            block_length=cfg.bootstrap_block_length,
            seed=cfg.bootstrap_seed,
            risk_free_per_period=cfg.risk_free_per_period,
        )
    except Exception as exc:
        raise IntegrationError(
            f"bootstrap-validation integration failed: {exc}"
        ) from exc

    return IntegrationResult(
        symbol=symbol,
        interval=interval,
        candle_count=len(candles),
        feature_rows=len(generated),
        feature_columns=columns,
        backtest=backtest,
        statistical_validation=statistical,
        bootstrap_validation=bootstrap,
    )


__all__ = [
    "IntegrationConfig",
    "IntegrationError",
    "IntegrationResult",
    "run_integrated_research",
]
