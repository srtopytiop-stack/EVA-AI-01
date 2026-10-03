"""Phase-15 tests for EVA-AI-01 Monitoring & Diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.monitoring import (
    MonitoringConfig,
    MonitoringError,
    MonitoringStatus,
    monitor_integration_result,
)


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
    trade_count: int = 4
    winning_trades: int = 3
    losing_trades: int = 1


@dataclass(frozen=True)
class Statistical:
    periodic_sharpe: float = 0.55
    deflated_sharpe_ratio: float = 0.10


@dataclass(frozen=True)
class Bootstrap:
    observations: int
    bootstrap_replications: int = 200
    block_length: int = 5
    bootstrap_positive_sharpe_fraction: float = 0.60
    sharpe_ci_lower: float = -0.2
    sharpe_ci_upper: float = 0.9


@dataclass(frozen=True)
class Trade:
    entry_timestamp: int
    exit_timestamp: int


@dataclass(frozen=True)
class Event:
    timestamp: int


@dataclass(frozen=True)
class Backtest:
    equity_curve: tuple[Point, ...]
    metrics: Performance = Performance()
    rejected_actions: int = 0
    trades: tuple[Trade, ...] = ()
    events: tuple[Event, ...] = ()
    bars_processed: int = 40


@dataclass(frozen=True)
class Result:
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


def good_result() -> Result:
    equity = 1000.0
    points: list[Point] = []
    returns = (0.004, -0.002, 0.003, -0.001, 0.002)

    for index in range(40):
        equity *= 1.0 + returns[index % len(returns)]
        points.append(Point(1_700_000_000_000 + index * 60_000, equity))

    return Result(
        symbol="BTCUSDT",
        interval="1m",
        candle_count=40,
        feature_rows=40,
        feature_columns=("return", "volatility", "volume_ratio"),
        backtest=Backtest(
            equity_curve=tuple(points),
            metrics=Performance(final_equity=equity),
            trades=(Trade(1_700_000_060_000, 1_700_000_180_000),),
            events=(
                Event(1_700_000_060_000),
                Event(1_700_000_120_000),
                Event(1_700_000_180_000),
            ),
            bars_processed=40,
        ),
        statistical_validation=Statistical(),
        bootstrap_validation=Bootstrap(observations=39),
    )


def test_healthy_result_passes() -> None:
    report = monitor_integration_result(good_result())
    assert report.status is MonitoringStatus.HEALTHY
    assert report.healthy is True
    assert report.issues == ()
    assert report.metrics["equity_points"] == 40
    assert report.metrics["bootstrap_observations"] == 39


def test_nan_equity_is_critical() -> None:
    result = good_result()
    points = list(result.backtest.equity_curve)
    points[10] = Point(points[10].timestamp, float("nan"))
    broken = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        Backtest(
            equity_curve=tuple(points),
            metrics=result.backtest.metrics,
            rejected_actions=result.backtest.rejected_actions,
            trades=result.backtest.trades,
            events=result.backtest.events,
            bars_processed=result.backtest.bars_processed,
        ),
        result.statistical_validation,
        result.bootstrap_validation,
    )
    report = monitor_integration_result(broken)
    assert report.status is MonitoringStatus.CRITICAL
    assert any(issue.code == "EQUITY_VALUE_NONFINITE" for issue in report.issues)


def test_final_equity_mismatch_is_critical() -> None:
    result = good_result()

    @dataclass(frozen=True)
    class MismatchResult(Result):
        @property
        def final_equity(self) -> float:
            return super().final_equity + 10.0

    changed = MismatchResult(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        result.backtest,
        result.statistical_validation,
        result.bootstrap_validation,
    )
    report = monitor_integration_result(changed)
    assert report.status is MonitoringStatus.CRITICAL
    assert any(issue.code == "FINAL_EQUITY_MISMATCH" for issue in report.issues)


def test_bootstrap_observation_mismatch_is_critical() -> None:
    result = good_result()
    broken = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        result.backtest,
        result.statistical_validation,
        Bootstrap(observations=38),
    )
    report = monitor_integration_result(broken)
    assert report.status is MonitoringStatus.CRITICAL
    assert any(issue.code == "BOOTSTRAP_OBSERVATION_MISMATCH" for issue in report.issues)


def test_invalid_bootstrap_probability_is_critical() -> None:
    result = good_result()
    broken = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        result.backtest,
        result.statistical_validation,
        Bootstrap(observations=39, bootstrap_positive_sharpe_fraction=1.5),
    )
    report = monitor_integration_result(broken)
    assert report.status is MonitoringStatus.CRITICAL
    assert any(issue.code == "BOOTSTRAP_PROBABILITY_INVALID" for issue in report.issues)


def test_trade_order_violation_is_critical() -> None:
    result = good_result()
    broken = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        Backtest(
            equity_curve=result.backtest.equity_curve,
            metrics=result.backtest.metrics,
            trades=(Trade(200, 100),),
            events=result.backtest.events,
        ),
        result.statistical_validation,
        result.bootstrap_validation,
    )
    report = monitor_integration_result(broken)
    assert report.status is MonitoringStatus.CRITICAL
    assert any(issue.code == "TRADE_TIME_ORDER_INVALID" for issue in report.issues)


def test_event_order_violation_is_critical() -> None:
    result = good_result()
    broken = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        Backtest(
            equity_curve=result.backtest.equity_curve,
            metrics=result.backtest.metrics,
            events=(Event(200), Event(100)),
        ),
        result.statistical_validation,
        result.bootstrap_validation,
    )
    report = monitor_integration_result(broken)
    assert report.status is MonitoringStatus.CRITICAL
    assert any(issue.code == "EVENT_TIME_ORDER_INVALID" for issue in report.issues)


def test_bad_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        MonitoringConfig(min_equity_points=1)


def test_strict_health_check_raises() -> None:
    result = good_result()
    points = list(result.backtest.equity_curve)
    points[2] = Point(points[2].timestamp, float("inf"))
    broken = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        Backtest(equity_curve=tuple(points), metrics=result.backtest.metrics),
        result.statistical_validation,
        result.bootstrap_validation,
    )
    report = monitor_integration_result(broken)
    with pytest.raises(MonitoringError):
        report.raise_if_unhealthy()
