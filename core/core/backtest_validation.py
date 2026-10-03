"""
EVA-AI-01 - Backtest Statistical Validation Bridge
==================================================
Phase 12-B: integration bridge between the Phase 11 BacktestResult and
Phase 12 statistical validation.

This module is deliberately post-backtest only. It does not generate
signals, mutate the portfolio, execute orders, or modify the original
BacktestResult.

The bridge is structural: it consumes an object exposing ``equity_curve``
instead of importing BacktestResult directly. This prevents a dependency
cycle between the orchestration layer and the statistical layer.
"""

from __future__ import annotations

from math import isfinite

from core.statistical_validation import (
    StatisticalValidationError,
    StatisticalValidationReport,
    validate_equity_curve,
)


class BacktestValidationError(ValueError):
    """Raised when a completed backtest cannot enter Phase 12 validation."""


def validate_backtest_result(
    backtest_result: object,
    *,
    number_of_trials: int,
    benchmark_sharpe: float = 0.0,
) -> StatisticalValidationReport:
    """
    Validate the equity curve contained in a completed Phase 11 result.

    ``number_of_trials`` is the number of materially distinct research
    choices searched before selecting the reported result. It is NOT inferred
    from trade count, candle count, or order count.

    ``benchmark_sharpe`` uses the same native return periodicity as the
    equity-curve returns consumed by Phase 12.
    """
    if backtest_result is None:
        raise BacktestValidationError("backtest_result cannot be None")

    try:
        equity_curve = getattr(backtest_result, "equity_curve")
    except AttributeError as exc:
        raise BacktestValidationError(
            "backtest_result must expose an equity_curve attribute"
        ) from exc

    if equity_curve is None:
        raise BacktestValidationError(
            "backtest_result.equity_curve cannot be None"
        )

    if isinstance(number_of_trials, bool) or not isinstance(number_of_trials, int):
        raise BacktestValidationError(
            "number_of_trials must be an integer"
        )
    if number_of_trials < 2:
        raise BacktestValidationError(
            "number_of_trials must be >= 2 for DSR"
        )

    if isinstance(benchmark_sharpe, bool):
        raise BacktestValidationError(
            "benchmark_sharpe must be numeric"
        )
    try:
        benchmark = float(benchmark_sharpe)
    except (TypeError, ValueError) as exc:
        raise BacktestValidationError(
            "benchmark_sharpe must be numeric"
        ) from exc
    if not isfinite(benchmark):
        raise BacktestValidationError(
            "benchmark_sharpe must be finite"
        )

    try:
        return validate_equity_curve(
            equity_curve,
            number_of_trials=number_of_trials,
            benchmark_sharpe=benchmark,
        )
    except StatisticalValidationError:
        # Preserve the statistical layer's domain error rather than hiding it
        # behind a generic integration exception.
        raise


__all__ = [
    "BacktestValidationError",
    "validate_backtest_result",
]
