"""Phase-14 production-gate tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.production_gate import (
    GateStatus,
    ProductionGateBlocked,
    ProductionGateConfig,
    audit_signal_execution_order,
    evaluate_integration_result,
    run_gated_research,
)


@dataclass(frozen=True)
class Point:
    timestamp: int
    equity: float


@dataclass(frozen=True)
class Backtest:
    bars_processed: int
    equity_curve: tuple[Point, ...]
    metrics: dict
    events: tuple = ()


@dataclass(frozen=True)
class Validation:
    periodic_sharpe_ratio: float = 0.7
    probabilistic_sharpe_ratio: float = 0.8
    deflated_sharpe_ratio: float = 0.75

    def to_dict(self) -> dict:
        return {
            "periodic_sharpe_ratio": self.periodic_sharpe_ratio,
            "probabilistic_sharpe_ratio": self.probabilistic_sharpe_ratio,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
        }


@dataclass(frozen=True)
class BootstrapValidation:
    observations: int
    bootstrap_replications: int = 200
    block_length: int = 5
    observed_sharpe: float = 0.7
    probabilistic_sharpe: float = 0.8
    deflated_sharpe: float = 0.75

    def to_dict(self) -> dict:
        return {
            "observations": self.observations,
            "bootstrap_replications": self.bootstrap_replications,
            "block_length": self.block_length,
            "observed_sharpe": self.observed_sharpe,
            "probabilistic_sharpe": self.probabilistic_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
        }


@dataclass(frozen=True)
class Result:
    symbol: str
    interval: str
    candle_count: int
    feature_rows: int
    feature_columns: tuple[str, ...]
    backtest: Backtest
    statistical_validation: Validation
    bootstrap_validation: BootstrapValidation

    @property
    def final_equity(self) -> float:
        return self.backtest.equity_curve[-1].equity


def _result(n: int = 60) -> Result:
    pattern = (0.004, -0.002, 0.003, -0.001, 0.002)
    equity = 1000.0
    points = []
    ts = 1_700_000_000_000
    for i in range(n):
        if i > 0:
            equity *= 1.0 + pattern[(i - 1) % len(pattern)]
        points.append(Point(timestamp=ts + i * 60_000, equity=equity))

    return Result(
        symbol="BTCUSDT",
        interval="1m",
        candle_count=n,
        feature_rows=n,
        feature_columns=("close", "return", "volatility"),
        backtest=Backtest(
            bars_processed=n,
            equity_curve=tuple(points),
            metrics={"final_equity": equity, "return": equity / 1000.0 - 1.0},
        ),
        statistical_validation=Validation(),
        bootstrap_validation=BootstrapValidation(observations=n - 1),
    )


def test_healthy_result_passes_without_performance_bias():
    report = evaluate_integration_result(_result())

    assert report.approved is True
    assert report.critical_failures == ()
    assert any(check.name == "equity_curve_integrity" and check.status is GateStatus.PASS for check in report.checks)


def test_negative_performance_does_not_become_a_gate_failure():
    result = _result()
    points = list(result.backtest.equity_curve)
    degraded = []
    for i, point in enumerate(points):
        negative_pattern = (0.998, 1.0015, 0.997, 1.0005, 0.999)
        degraded_equity = 1000.0
        for j in range(i):
            degraded_equity *= negative_pattern[j % len(negative_pattern)]
        degraded.append(Point(timestamp=point.timestamp, equity=degraded_equity))

    changed = Result(
        symbol=result.symbol,
        interval=result.interval,
        candle_count=result.candle_count,
        feature_rows=result.feature_rows,
        feature_columns=result.feature_columns,
        backtest=Backtest(
            bars_processed=result.backtest.bars_processed,
            equity_curve=tuple(degraded),
            metrics={"final_equity": degraded[-1].equity},
        ),
        statistical_validation=result.statistical_validation,
        bootstrap_validation=result.bootstrap_validation,
    )

    report = evaluate_integration_result(changed)
    assert report.approved is True


def test_nonfinite_equity_is_blocked():
    result = _result()
    points = list(result.backtest.equity_curve)
    points[25] = Point(points[25].timestamp, float("nan"))
    changed = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        Backtest(result.backtest.bars_processed, tuple(points), result.backtest.metrics),
        result.statistical_validation,
        result.bootstrap_validation,
    )

    report = evaluate_integration_result(changed)
    assert report.approved is False
    assert any("equity_curve_integrity" in failure for failure in report.critical_failures)


def test_bootstrap_observation_mismatch_is_blocked():
    result = _result()
    changed = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        result.backtest,
        result.statistical_validation,
        BootstrapValidation(observations=result.bootstrap_validation.observations - 1),
    )

    report = evaluate_integration_result(changed)
    assert report.approved is False
    assert any("bootstrap_validation" in failure for failure in report.critical_failures)


def test_insufficient_bootstrap_replications_are_blocked():
    result = _result()
    changed = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        result.backtest,
        result.statistical_validation,
        BootstrapValidation(observations=result.bootstrap_validation.observations, bootstrap_replications=199),
    )

    report = evaluate_integration_result(changed)
    assert report.approved is False


def test_temporal_audit_requires_strictly_future_execution():
    good = audit_signal_execution_order(((1, 2), (2, 5), (10, 11)))
    assert good.passed is True
    assert good.audited_pairs == 3

    bad = audit_signal_execution_order(((1, 1), (2, 1), (10, 12)))
    assert bad.passed is False
    assert len(bad.violations) == 2


def test_strict_temporal_mode_blocks_when_audit_data_is_unavailable():
    result = _result()
    report = evaluate_integration_result(
        result,
        config=ProductionGateConfig(require_temporal_audit=True),
    )
    assert report.approved is False


def test_raise_if_blocked_raises():
    result = _result()
    changed = Result(
        result.symbol,
        result.interval,
        result.candle_count,
        result.feature_rows,
        result.feature_columns,
        result.backtest,
        result.statistical_validation,
        BootstrapValidation(observations=result.bootstrap_validation.observations - 1),
    )

    report = evaluate_integration_result(changed)
    with pytest.raises(ProductionGateBlocked):
        report.raise_if_blocked()


def test_run_gated_research_wraps_existing_integration(monkeypatch):
    result = _result()

    def fake_run(market_data, config=None):
        assert market_data == "MARKET"
        assert config == "CONFIG"
        return result

    import sys
    import types

    integration = types.ModuleType("core.system_integration")
    integration.run_integrated_research = fake_run
    monkeypatch.setitem(sys.modules, "core.system_integration", integration)

    returned_result, report = run_gated_research(
        "MARKET",
        integration_config="CONFIG",
    )

    assert returned_result is result
    assert report.approved is True
