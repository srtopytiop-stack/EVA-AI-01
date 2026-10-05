"""Phase 12-C moving-block bootstrap validation tests."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from core.backtest_metrics import EquityPoint
from core.bootstrap_validation import (
    BootstrapValidationError,
    validate_backtest_result_with_bootstrap,
    validate_returns_with_moving_block_bootstrap,
)


@dataclass(frozen=True)
class MockBacktestResult:
    equity_curve: tuple[EquityPoint, ...]
    trades: tuple[object, ...] = ()


def make_returns() -> list[float]:
    # Deliberately serial-looking synthetic data with enough observations for
    # a dependence-aware bootstrap test. Values remain > -1 so compounding is valid.
    pattern = (
        0.010,
        0.008,
        0.006,
        -0.004,
        -0.006,
        0.003,
        0.005,
        -0.002,
    )

    return [
        pattern[index % len(pattern)]
        for index in range(64)
    ]


def make_backtest_result() -> MockBacktestResult:
    returns = make_returns()
    equity = 1_000.0

    points = [
        EquityPoint(
            timestamp=1_800_000_000_000,
            equity=equity,
            cash=equity,
            gross_exposure=0.0,
            exposure_pct=0.0,
            drawdown_pct=0.0,
        )
    ]

    for index, value in enumerate(
        returns,
        start=1,
    ):
        equity *= 1.0 + value

        points.append(
            EquityPoint(
                timestamp=(
                    1_800_000_000_000
                    + index * 60_000
                ),
                equity=equity,
                cash=equity,
                gross_exposure=0.0,
                exposure_pct=0.0,
                drawdown_pct=0.0,
            )
        )

    return MockBacktestResult(
        equity_curve=tuple(points),
        trades=(object(),),
    )


def test_bootstrap_is_reproducible_for_same_seed() -> None:
    returns = make_returns()

    first = validate_returns_with_moving_block_bootstrap(
        returns,
        bootstrap_replications=250,
        block_length=4,
        seed=123,
    )

    second = validate_returns_with_moving_block_bootstrap(
        returns,
        bootstrap_replications=250,
        block_length=4,
        seed=123,
    )

    assert first == second


def test_bootstrap_preserves_observed_statistics_separately_from_resamples() -> None:
    returns = make_returns()

    report = validate_returns_with_moving_block_bootstrap(
        returns,
        bootstrap_replications=250,
        block_length=4,
        seed=123,
    )

    from core.statistical_validation import periodic_sharpe_ratio

    assert report.observations == len(returns)

    assert report.observed_periodic_sharpe == pytest.approx(
        periodic_sharpe_ratio(returns)
    )

    assert report.bootstrap_replications == 250

    assert (
        0.0
        <= report.bootstrap_positive_sharpe_fraction
        <= 1.0
    )

    assert report.sharpe_ci_lower <= report.sharpe_ci_upper
    assert (
        report.total_return_ci_lower
        <= report.total_return_ci_upper
    )
    assert (
        report.max_drawdown_ci_lower
        <= report.max_drawdown_ci_upper
    )


def test_bootstrap_does_not_use_trade_count_as_research_trial_count() -> None:
    result = make_backtest_result()

    report = validate_backtest_result_with_bootstrap(
        result,
        bootstrap_replications=250,
        block_length=4,
        seed=7,
    )

    assert report.bootstrap_replications == 250
    assert result.trades == (
        result.trades[0],
    )


def test_backtest_result_bridge_uses_only_equity_curve() -> None:
    result = make_backtest_result()
    before = tuple(result.equity_curve)

    report = validate_backtest_result_with_bootstrap(
        result,
        bootstrap_replications=250,
        block_length=4,
        seed=42,
    )

    assert (
        report.observations
        == len(result.equity_curve) - 1
    )

    assert tuple(result.equity_curve) == before


def test_report_is_json_safe() -> None:
    report = validate_returns_with_moving_block_bootstrap(
        make_returns(),
        bootstrap_replications=250,
        block_length=4,
        seed=42,
    )

    json.dumps(
        report.to_dict(),
        allow_nan=False,
    )


def test_invalid_replication_count_is_rejected() -> None:
    with pytest.raises(BootstrapValidationError):
        validate_returns_with_moving_block_bootstrap(
            make_returns(),
            bootstrap_replications=199,
            block_length=4,
        )


def test_invalid_block_length_is_rejected() -> None:
    returns = make_returns()

    with pytest.raises(BootstrapValidationError):
        validate_returns_with_moving_block_bootstrap(
            returns,
            bootstrap_replications=250,
            block_length=1,
        )

    with pytest.raises(BootstrapValidationError):
        validate_returns_with_moving_block_bootstrap(
            returns,
            bootstrap_replications=250,
            block_length=len(returns),
        )


def test_too_short_series_is_rejected() -> None:
    with pytest.raises(BootstrapValidationError):
        validate_returns_with_moving_block_bootstrap(
            [0.01] * 10,
            bootstrap_replications=250,
            block_length=2,
        )


def test_invalid_backtest_result_is_rejected() -> None:
    with pytest.raises(BootstrapValidationError):
        validate_backtest_result_with_bootstrap(
            object(),
            bootstrap_replications=250,
            block_length=4,
        )


def test_zero_variance_returns_are_rejected_as_bootstrap_error() -> None:
    """
    Bootstrap must translate undefined Sharpe inference
    into its own domain-specific error.
    """
    with pytest.raises(
        BootstrapValidationError
    ) as exc_info:
        validate_returns_with_moving_block_bootstrap(
            [0.0] * 64,
            bootstrap_replications=250,
            block_length=4,
            seed=42,
        )

    assert (
        "return variance is zero"
        in str(exc_info.value)
    )
