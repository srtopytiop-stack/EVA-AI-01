"""Phase-15-B tests for the combined research/gate/monitoring boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from core.monitoring import MonitoringStatus
from core.phase15b_runtime import (
    Phase15BConfig,
    Phase15BRuntimeError,
    Phase15BStatus,
    diagnose_integrated_result,
    run_phase15b,
)
from core.production_gate import ProductionGateConfig


@dataclass(frozen=True)
class Point:
    timestamp: int
    equity: float


@dataclass(frozen=True)
class Performance:
    initial_equity: float = 1000.0
    final_equity: float = 1100.0
    total_return_pct: float = 10.0
    max_drawdown_pct: float = -0.08
    bars_processed: int = 40
    rejected_actions: int = 0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


@dataclass(frozen=True)
class Statistical:
    periodic_sharpe: float = 0.55
    probabilistic_sharpe: float = 0.60
    deflated_sharpe_ratio: float = 0.10

    def to_dict(self) -> dict[str, float]:
        return {
            "periodic_sharpe": self.periodic_sharpe,
            "probabilistic_sharpe": self.probabilistic_sharpe,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
        }


@dataclass(frozen=True)
class Bootstrap:
    observations: int = 39
    bootstrap_replications: int = 200
    block_length: int = 5
    bootstrap_positive_sharpe_fraction: float = 0.60
    sharpe_ci_lower: float = -0.20
    sharpe_ci_upper: float = 0.90

    def to_dict(self) -> dict[str, float | int]:
        return {
            "observations": self.observations,
            "bootstrap_replications": self.bootstrap_replications,
            "block_length": self.block_length,
            "bootstrap_positive_sharpe_fraction": self.bootstrap_positive_sharpe_fraction,
            "sharpe_ci_lower": self.sharpe_ci_lower,
            "sharpe_ci_upper": self.sharpe_ci_upper,
        }


@dataclass(frozen=True)
class Backtest:
    equity_curve: tuple[Point, ...]
    metrics: Performance
    rejected_actions: int = 0
    trades: tuple = ()
    events: tuple = ()
    bars_processed: int = 40


@dataclass(frozen=True)
class Research:
    symbol: str
    interval: str
    candle_count: int
    feature_rows: int
    feature_columns: tuple[str, ...]
    backtest: Backtest
    statistical_validation: Statistical
    bootstrap_validation: Bootstrap

    @property
    def final_equity(self) -> float:
        return self.backtest.equity_curve[-1].equity

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "candle_count": self.candle_count,
            "feature_rows": self.feature_rows,
            "feature_columns": list(self.feature_columns),
            "backtest": {
                "bars_processed": self.backtest.bars_processed,
                "rejected_actions": self.backtest.rejected_actions,
                "equity_curve": [
                    {"timestamp": p.timestamp, "equity": p.equity}
                    for p in self.backtest.equity_curve
                ],
                "metrics": self.backtest.metrics.__dict__,
            },
            "statistical_validation": self.statistical_validation.to_dict(),
            "bootstrap_validation": self.bootstrap_validation.to_dict(),
            "final_equity": self.final_equity,
        }


def good_result() -> Research:
    returns = (0.004, -0.002, 0.003, -0.001, 0.002)
    equity = 1000.0
    points: list[Point] = []

    for index in range(40):
        equity *= 1.0 + returns[index % len(returns)]
        points.append(Point(1_700_000_000_000 + index * 60_000, equity))

    metrics = Performance(
        final_equity=equity,
        initial_equity=1000.0,
    )

    return Research(
        symbol="BTCUSDT",
        interval="1m",
        candle_count=40,
        feature_rows=40,
        feature_columns=("return", "volatility", "volume_ratio"),
        backtest=Backtest(
            equity_curve=tuple(points),
            metrics=metrics,
            bars_processed=40,
        ),
        statistical_validation=Statistical(),
        bootstrap_validation=Bootstrap(observations=39),
    )


def test_diagnose_integrated_result_ready() -> None:
    result = diagnose_integrated_result(good_result())

    assert result.status is Phase15BStatus.READY
    assert result.production_gate.approved is True
    assert result.monitoring.status is MonitoringStatus.HEALTHY
    assert result.release_ready is True
    assert result.to_dict()["release_ready"] is True


def test_gate_block_is_visible_without_hiding_monitoring() -> None:
    result = diagnose_integrated_result(
        good_result(),
        gate_config=ProductionGateConfig(min_equity_points=100),
    )

    assert result.production_gate.approved is False
    assert result.monitoring.status is MonitoringStatus.HEALTHY
    assert result.status is Phase15BStatus.GATE_BLOCKED
    assert result.release_ready is False


def test_monitoring_critical_state_is_visible() -> None:
    base = good_result()
    points = list(base.backtest.equity_curve)
    points[10] = Point(points[10].timestamp, float("nan"))

    broken = replace(
        base,
        backtest=replace(
            base.backtest,
            equity_curve=tuple(points),
        ),
    )

    result = diagnose_integrated_result(broken)

    assert result.monitoring.status is MonitoringStatus.CRITICAL
    assert result.release_ready is False
    assert result.status in {
        Phase15BStatus.MONITORING_CRITICAL,
        Phase15BStatus.GATE_BLOCKED_AND_MONITORING_CRITICAL,
    }


def test_strict_mode_raises() -> None:
    with pytest.raises(Phase15BRuntimeError):
        # The function is monkeypatched at the module boundary below.
        import core.phase15b_runtime as runtime

        original = runtime.run_integrated_research
        try:
            runtime.run_integrated_research = lambda market_data, config=None: good_result()  # type: ignore[assignment]
            run_phase15b(
                "MARKET",
                config=Phase15BConfig(
                    strict=True,
                    gate_config=ProductionGateConfig(min_equity_points=100),
                ),
            )
        finally:
            runtime.run_integrated_research = original  # type: ignore[assignment]
