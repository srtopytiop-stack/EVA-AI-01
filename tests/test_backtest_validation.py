"""Phase 12-B tests for the Phase 11 -> Phase 12 validation bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from core.backtest_metrics import EquityPoint
from core.backtest_validation import (
    BacktestValidationError,
    validate_backtest_result,
)
from core.statistical_validation import StatisticalValidationError


@dataclass(frozen=True)
class MockBacktestResult:
    symbol: str
    equity_curve: tuple[EquityPoint, ...]
    trades: tuple[object, ...] = ()


def make_result() -> MockBacktestResult:
    base = 1_800_000_000_000
    values = (
        1_000.0,
        1_006.0,
        1_001.0,
        1_010.0,
        1_014.0,
        1_020.0,
        1_015.0,
        1_026.0,
        1_030.0,
        1_037.0,
        1_034.0,
        1_045.0,
    )

    return MockBacktestResult(
        symbol="BTCUSDT",
        equity_curve=tuple(
            EquityPoint(
                timestamp=base + index * 60_000,
                equity=value,
                cash=value,
                gross_exposure=0.0,
                exposure_pct=0.0,
                drawdown_pct=0.0,
            )
            for index, value in enumerate(values)
        ),
    )


def test_completed_backtest_can_enter_phase12() -> None:
    result = make_result()

    report = validate_backtest_result(
        result,
        number_of_trials=25,
    )

    assert report.number_of_trials == 25
    assert report.distribution.observations == len(result.equity_curve) - 1
    assert 0.0 <= report.probabilistic_sharpe <= 1.0
    assert 0.0 <= report.deflated_sharpe <= 1.0


def test_trial_count_is_explicit_and_not_trade_count() -> None:
    result = MockBacktestResult(
        symbol="BTCUSDT",
        equity_curve=make_result().equity_curve,
        trades=(object(),),
    )

    report = validate_backtest_result(
        result,
        number_of_trials=37,
    )

    assert report.number_of_trials == 37


def test_original_backtest_result_is_not_mutated() -> None:
    result = make_result()
    before = tuple(result.equity_curve)

    _ = validate_backtest_result(
        result,
        number_of_trials=25,
    )

    assert tuple(result.equity_curve) == before


def test_benchmark_sharpe_is_forwarded() -> None:
    result = make_result()

    zero = validate_backtest_result(
        result,
        number_of_trials=10,
        benchmark_sharpe=0.0,
    )
    positive = validate_backtest_result(
        result,
        number_of_trials=10,
        benchmark_sharpe=0.1,
    )

    assert zero.benchmark_sharpe == pytest.approx(0.0)
    assert positive.benchmark_sharpe == pytest.approx(0.1)
    assert positive.probabilistic_sharpe < zero.probabilistic_sharpe


def test_report_is_json_safe() -> None:
    report = validate_backtest_result(
        make_result(),
        number_of_trials=25,
    )

    json.dumps(report.to_dict(), allow_nan=False)


def test_missing_equity_curve_is_rejected() -> None:
    with pytest.raises(BacktestValidationError):
        validate_backtest_result(
            object(),
            number_of_trials=25,
        )


def test_invalid_trial_count_is_rejected() -> None:
    result = make_result()

    with pytest.raises(BacktestValidationError):
        validate_backtest_result(
            result,
            number_of_trials=1,
        )

    with pytest.raises(BacktestValidationError):
        validate_backtest_result(
            result,
            number_of_trials=True,
        )


def test_nonfinite_benchmark_is_rejected() -> None:
    with pytest.raises(BacktestValidationError):
        validate_backtest_result(
            make_result(),
            number_of_trials=25,
            benchmark_sharpe=float("nan"),
        )


def test_statistical_validation_errors_are_not_hidden() -> None:
    result = MockBacktestResult(
        symbol="BTCUSDT",
        equity_curve=tuple(
            EquityPoint(
                timestamp=1_800_000_000_000 + index * 60_000,
                equity=1_000.0 + index * 10.0,
                cash=1_000.0 + index * 10.0,
                gross_exposure=0.0,
                exposure_pct=0.0,
                drawdown_pct=0.0,
            )
            for index in range(4)
        ),
    )

    # Four points create only three returns, below Phase 12's minimum
    # observation count. The bridge must preserve the statistical error.
    with pytest.raises(StatisticalValidationError):
        validate_backtest_result(
            result,
            number_of_trials=25,
        )
