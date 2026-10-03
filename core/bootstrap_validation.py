"""
EVA-AI-01 - Dependence-Aware Bootstrap Validation
==================================================
Phase 12-C: Moving-Block Bootstrap robustness validation.

Purpose
-------
Phase 12-A provides PSR/DSR inference for a completed backtest and
Phase 12-B provides the BacktestResult -> statistical-validation bridge.
Phase 12-C adds a non-parametric, dependence-aware resampling layer.

The ordinary iid bootstrap resamples individual observations independently.
That can destroy short-range temporal dependence in a time series. A
moving-block bootstrap (MBB) instead samples contiguous blocks, preserving
local serial structure without fitting a parametric return model.

Critical design rules
---------------------
- This module is post-backtest analytics only.
- It never generates signals, executes orders, or mutates a portfolio.
- It never reads future market data outside the supplied completed return
  series/equity curve.
- ``bootstrap_replications`` is the number of Monte-Carlo resamples. It is
  NOT the number of strategy/research trials used by DSR.
- ``block_length`` is a resampling hyperparameter and is recorded in the
  report so the experiment remains reproducible and auditable.

The bootstrap output is an empirical distribution of statistics, not a new
trading strategy and not a guarantee of out-of-sample performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log1p, sqrt
from random import Random
from statistics import fmean
from typing import Sequence

from core.statistical_validation import (
    StatisticalValidationError,
    periodic_sharpe_ratio,
    returns_from_equity_curve,
)

MIN_BOOTSTRAP_OBSERVATIONS = 20
MIN_BOOTSTRAP_REPLICATIONS = 200
MIN_BLOCK_LENGTH = 2
EPS = 1e-12


class BootstrapValidationError(ValueError):
    """Raised when bootstrap-validation inputs are invalid."""


@dataclass(frozen=True)
class BootstrapValidationReport:
    """Empirical uncertainty summary produced by moving-block bootstrap."""

    observations: int
    bootstrap_replications: int
    block_length: int
    seed: int | None
    risk_free_per_period: float

    observed_periodic_sharpe: float
    bootstrap_mean_sharpe: float
    bootstrap_std_sharpe: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    bootstrap_positive_sharpe_fraction: float

    observed_total_return: float
    bootstrap_mean_total_return: float
    bootstrap_std_total_return: float
    total_return_ci_lower: float
    total_return_ci_upper: float

    observed_max_drawdown: float
    bootstrap_mean_max_drawdown: float
    bootstrap_std_max_drawdown: float
    max_drawdown_ci_lower: float
    max_drawdown_ci_upper: float

    def to_dict(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "bootstrap_replications": self.bootstrap_replications,
            "block_length": self.block_length,
            "seed": self.seed,
            "risk_free_per_period": self.risk_free_per_period,
            "observed_periodic_sharpe": self.observed_periodic_sharpe,
            "bootstrap_mean_sharpe": self.bootstrap_mean_sharpe,
            "bootstrap_std_sharpe": self.bootstrap_std_sharpe,
            "sharpe_ci_lower": self.sharpe_ci_lower,
            "sharpe_ci_upper": self.sharpe_ci_upper,
            "bootstrap_positive_sharpe_fraction": self.bootstrap_positive_sharpe_fraction,
            "observed_total_return": self.observed_total_return,
            "bootstrap_mean_total_return": self.bootstrap_mean_total_return,
            "bootstrap_std_total_return": self.bootstrap_std_total_return,
            "total_return_ci_lower": self.total_return_ci_lower,
            "total_return_ci_upper": self.total_return_ci_upper,
            "observed_max_drawdown": self.observed_max_drawdown,
            "bootstrap_mean_max_drawdown": self.bootstrap_mean_max_drawdown,
            "bootstrap_std_max_drawdown": self.bootstrap_std_max_drawdown,
            "max_drawdown_ci_lower": self.max_drawdown_ci_lower,
            "max_drawdown_ci_upper": self.max_drawdown_ci_upper,
        }


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise BootstrapValidationError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise BootstrapValidationError(f"{name} must be numeric") from exc
    if not isfinite(normalized):
        raise BootstrapValidationError(f"{name} must be finite")
    return normalized


def _normalise_returns(returns: Sequence[float]) -> list[float]:
    if isinstance(returns, (str, bytes)):
        raise BootstrapValidationError("returns must be a numeric sequence")

    try:
        values = [
            _finite_float(value, f"return at index {index}")
            for index, value in enumerate(returns)
        ]
    except TypeError as exc:
        raise BootstrapValidationError("returns must be a numeric sequence") from exc

    if len(values) < MIN_BOOTSTRAP_OBSERVATIONS:
        raise BootstrapValidationError(
            f"at least {MIN_BOOTSTRAP_OBSERVATIONS} return observations are required"
        )

    for index, value in enumerate(values):
        if value <= -1.0:
            raise BootstrapValidationError(
                f"return at index {index} must be greater than -1"
            )

    return values


def _validate_configuration(
    *,
    observations: int,
    bootstrap_replications: int,
    block_length: int,
    seed: int | None,
    risk_free_per_period: float,
) -> float:
    if isinstance(bootstrap_replications, bool) or not isinstance(
        bootstrap_replications, int
    ):
        raise BootstrapValidationError(
            "bootstrap_replications must be an integer"
        )
    if bootstrap_replications < MIN_BOOTSTRAP_REPLICATIONS:
        raise BootstrapValidationError(
            f"bootstrap_replications must be >= {MIN_BOOTSTRAP_REPLICATIONS}"
        )

    if isinstance(block_length, bool) or not isinstance(block_length, int):
        raise BootstrapValidationError("block_length must be an integer")
    if block_length < MIN_BLOCK_LENGTH:
        raise BootstrapValidationError(
            f"block_length must be >= {MIN_BLOCK_LENGTH}"
        )
    if block_length >= observations:
        raise BootstrapValidationError(
            "block_length must be smaller than the number of observations"
        )

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise BootstrapValidationError("seed must be an integer or None")

    risk_free = _finite_float(
        risk_free_per_period,
        "risk_free_per_period",
    )
    if risk_free <= -1.0:
        raise BootstrapValidationError("risk_free_per_period must be > -1")

    return risk_free


def moving_block_bootstrap_sample(
    returns: Sequence[float],
    *,
    block_length: int,
    rng: Random,
) -> list[float]:
    """
    Generate one moving-block-bootstrap sample with the same length as input.

    Every selected block is a contiguous slice of the original chronological
    return series. Blocks are sampled with replacement and concatenated until
    the target length is reached.
    """
    values = _normalise_returns(returns)
    if not isinstance(rng, Random):
        raise BootstrapValidationError("rng must be an instance of random.Random")
    _validate_configuration(
        observations=len(values),
        bootstrap_replications=MIN_BOOTSTRAP_REPLICATIONS,
        block_length=block_length,
        seed=0,
        risk_free_per_period=0.0,
    )

    sample: list[float] = []
    max_start = len(values) - block_length

    while len(sample) < len(values):
        start = rng.randint(0, max_start)
        stop = min(start + block_length, len(values))
        sample.extend(values[start:stop])

    return sample[: len(values)]


def _wealth_path(returns: Sequence[float]) -> list[float]:
    """Build a unit-start wealth path from simple returns."""
    wealth = 1.0
    path = [wealth]

    for index, value in enumerate(returns):
        if value <= -1.0 or not isfinite(value):
            raise BootstrapValidationError(
                f"return at index {index} is invalid for compounding"
            )
        wealth *= 1.0 + value
        if not isfinite(wealth) or wealth <= 0.0:
            raise BootstrapValidationError(
                "bootstrap wealth path became non-finite or non-positive"
            )
        path.append(wealth)

    return path


def _total_return(returns: Sequence[float]) -> float:
    """Calculate compounded total return without silently accepting overflow."""
    log_total = 0.0
    for index, value in enumerate(returns):
        if value <= -1.0 or not isfinite(value):
            raise BootstrapValidationError(
                f"return at index {index} is invalid for compounding"
            )
        log_total += log1p(value)

    if log_total > 709.0:
        raise BootstrapValidationError(
            "compounded total return exceeds finite floating-point range"
        )

    return exp(log_total) - 1.0


def _max_drawdown(returns: Sequence[float]) -> float:
    """Calculate maximum drawdown from a simple-return path."""
    path = _wealth_path(returns)
    peak = path[0]
    maximum_drawdown = 0.0

    for wealth in path:
        if wealth > peak:
            peak = wealth
        drawdown = wealth / peak - 1.0
        if drawdown < maximum_drawdown:
            maximum_drawdown = drawdown

    return maximum_drawdown


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Linear-interpolated empirical quantile."""
    if not sorted_values:
        raise BootstrapValidationError("cannot calculate a quantile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise BootstrapValidationError("quantile probability must be in [0, 1]")

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        float(sorted_values[lower])
        + fraction * (float(sorted_values[upper]) - float(sorted_values[lower]))
    )


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = fmean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return sqrt(max(variance, 0.0))


def validate_returns_with_moving_block_bootstrap(
    returns: Sequence[float],
    *,
    bootstrap_replications: int = 2_000,
    block_length: int,
    seed: int | None = 42,
    risk_free_per_period: float = 0.0,
) -> BootstrapValidationReport:
    """
    Estimate empirical uncertainty for return statistics using MBB.

    The observed statistics are computed once from the supplied chronological
    returns. Bootstrap samples are generated only from those same returns.
    No strategy re-optimization occurs inside the bootstrap loop.
    """
    values = _normalise_returns(returns)
    risk_free = _validate_configuration(
        observations=len(values),
        bootstrap_replications=bootstrap_replications,
        block_length=block_length,
        seed=seed,
        risk_free_per_period=risk_free_per_period,
    )

    observed_sharpe = periodic_sharpe_ratio(
        values,
        risk_free_per_period=risk_free,
    )
    observed_total_return = _total_return(values)
    observed_max_drawdown = _max_drawdown(values)

    rng = Random(seed)
    bootstrap_sharpes: list[float] = []
    bootstrap_total_returns: list[float] = []
    bootstrap_drawdowns: list[float] = []

    for _ in range(bootstrap_replications):
        sample = moving_block_bootstrap_sample(
            values,
            block_length=block_length,
            rng=rng,
        )

        bootstrap_sharpes.append(
            periodic_sharpe_ratio(
                sample,
                risk_free_per_period=risk_free,
            )
        )
        bootstrap_total_returns.append(_total_return(sample))
        bootstrap_drawdowns.append(_max_drawdown(sample))

    sorted_sharpes = sorted(bootstrap_sharpes)
    sorted_total_returns = sorted(bootstrap_total_returns)
    sorted_drawdowns = sorted(bootstrap_drawdowns)

    return BootstrapValidationReport(
        observations=len(values),
        bootstrap_replications=bootstrap_replications,
        block_length=block_length,
        seed=seed,
        risk_free_per_period=risk_free,
        observed_periodic_sharpe=observed_sharpe,
        bootstrap_mean_sharpe=fmean(bootstrap_sharpes),
        bootstrap_std_sharpe=_sample_std(bootstrap_sharpes),
        sharpe_ci_lower=_quantile(sorted_sharpes, 0.025),
        sharpe_ci_upper=_quantile(sorted_sharpes, 0.975),
        bootstrap_positive_sharpe_fraction=sum(
            value > 0.0 for value in bootstrap_sharpes
        )
        / bootstrap_replications,
        observed_total_return=observed_total_return,
        bootstrap_mean_total_return=fmean(bootstrap_total_returns),
        bootstrap_std_total_return=_sample_std(bootstrap_total_returns),
        total_return_ci_lower=_quantile(sorted_total_returns, 0.025),
        total_return_ci_upper=_quantile(sorted_total_returns, 0.975),
        observed_max_drawdown=observed_max_drawdown,
        bootstrap_mean_max_drawdown=fmean(bootstrap_drawdowns),
        bootstrap_std_max_drawdown=_sample_std(bootstrap_drawdowns),
        max_drawdown_ci_lower=_quantile(sorted_drawdowns, 0.025),
        max_drawdown_ci_upper=_quantile(sorted_drawdowns, 0.975),
    )


def validate_backtest_result_with_bootstrap(
    backtest_result: object,
    *,
    bootstrap_replications: int = 2_000,
    block_length: int,
    seed: int | None = 42,
    risk_free_per_period: float = 0.0,
) -> BootstrapValidationReport:
    """
    Run Phase 12-C on a completed Phase 11 BacktestResult-like object.

    The bridge intentionally accesses only ``equity_curve`` so this module
    does not depend on the concrete BacktestResult class.
    """
    if backtest_result is None:
        raise BootstrapValidationError("backtest_result cannot be None")

    try:
        equity_curve = getattr(backtest_result, "equity_curve")
    except AttributeError as exc:
        raise BootstrapValidationError(
            "backtest_result must expose an equity_curve attribute"
        ) from exc

    if equity_curve is None:
        raise BootstrapValidationError(
            "backtest_result.equity_curve cannot be None"
        )

    try:
        returns = returns_from_equity_curve(equity_curve)
    except StatisticalValidationError as exc:
        raise BootstrapValidationError(
            "backtest_result.equity_curve failed statistical validation"
        ) from exc

    return validate_returns_with_moving_block_bootstrap(
        returns,
        bootstrap_replications=bootstrap_replications,
        block_length=block_length,
        seed=seed,
        risk_free_per_period=risk_free_per_period,
    )


__all__ = [
    "BootstrapValidationError",
    "BootstrapValidationReport",
    "MIN_BOOTSTRAP_OBSERVATIONS",
    "MIN_BOOTSTRAP_REPLICATIONS",
    "moving_block_bootstrap_sample",
    "validate_backtest_result_with_bootstrap",
    "validate_returns_with_moving_block_bootstrap",
]
