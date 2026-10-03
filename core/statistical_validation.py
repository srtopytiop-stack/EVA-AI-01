"""
EVA-AI-01 - Statistical Backtest Validation
============================================
Phase 12: inference-aware backtest validation.

This module is deliberately separate from the execution/backtest engines.
It performs post-backtest statistical inference only; it never generates
signals, mutates a portfolio, or changes an execution decision.

Implemented methods
-------------------
1. Return-distribution moments (mean, sample volatility, skewness,
   excess kurtosis).
2. Periodic Sharpe ratio.
3. Probabilistic Sharpe Ratio (PSR) for a threshold Sharpe ratio.
4. Deflated Sharpe Ratio (DSR) using a search-adjusted expected maximum
   Sharpe ratio under the null, following the Bailey/López de Prado
   framework.
5. Conversion from a Phase 11 equity curve to periodic returns.

Important statistical convention
---------------------------------
PSR/DSR inference is performed at the native return frequency. Therefore
`benchmark_sharpe` must use the same periodicity as the supplied returns.
Annualized Sharpe is reported separately for display only.

References
----------
Bailey & López de Prado (2012), The Sharpe Ratio Efficient Frontier.
Bailey & López de Prado (2014), The Deflated Sharpe Ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import e, isfinite, sqrt
from statistics import NormalDist, mean, median
from typing import Sequence

SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
EULER_MASCHERONI = 0.5772156649015329
EPS = 1e-12
MIN_OBSERVATIONS = 4


class StatisticalValidationError(ValueError):
    """Raised when statistical-validation inputs are invalid."""


@dataclass(frozen=True)
class ReturnDistribution:
    """Descriptive statistics of a periodic return series."""

    observations: int
    mean_return: float
    sample_std: float
    skewness: float
    excess_kurtosis: float
    minimum_return: float
    maximum_return: float

    @property
    def pearson_kurtosis(self) -> float:
        """Return kurtosis including the normal-distribution baseline of 3."""

        return self.excess_kurtosis + 3.0

    def to_dict(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "mean_return": self.mean_return,
            "sample_std": self.sample_std,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "pearson_kurtosis": self.pearson_kurtosis,
            "minimum_return": self.minimum_return,
            "maximum_return": self.maximum_return,
        }


@dataclass(frozen=True)
class StatisticalValidationReport:
    """Inference-aware summary of a completed return series."""

    distribution: ReturnDistribution
    periodic_sharpe: float
    annualized_sharpe: float
    benchmark_sharpe: float
    probabilistic_sharpe: float
    number_of_trials: int
    expected_max_sharpe: float
    deflated_sharpe: float
    periods_per_year: float

    def to_dict(self) -> dict[str, object]:
        return {
            "distribution": self.distribution.to_dict(),
            "periodic_sharpe": self.periodic_sharpe,
            "annualized_sharpe": self.annualized_sharpe,
            "benchmark_sharpe": self.benchmark_sharpe,
            "probabilistic_sharpe": self.probabilistic_sharpe,
            "number_of_trials": self.number_of_trials,
            "expected_max_sharpe": self.expected_max_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "periods_per_year": self.periods_per_year,
        }


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise StatisticalValidationError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise StatisticalValidationError(f"{name} must be numeric") from exc
    if not isfinite(normalized):
        raise StatisticalValidationError(f"{name} must be finite")
    return normalized


def _normalise_returns(returns: Sequence[float]) -> list[float]:
    if isinstance(returns, (str, bytes)):
        raise StatisticalValidationError("returns must be a numeric sequence")

    values: list[float] = []
    for index, value in enumerate(returns):
        normalized = _finite_float(value, f"return at index {index}")
        values.append(normalized)

    if len(values) < MIN_OBSERVATIONS:
        raise StatisticalValidationError(
            f"at least {MIN_OBSERVATIONS} return observations are required"
        )

    return values


def return_distribution(returns: Sequence[float]) -> ReturnDistribution:
    """Calculate moment statistics using central moments."""

    values = _normalise_returns(returns)
    n = len(values)
    average = mean(values)

    centered = [value - average for value in values]
    population_m2 = sum(value**2 for value in centered) / n

    if population_m2 <= EPS:
        raise StatisticalValidationError(
            "return variance is zero; Sharpe inference is undefined"
        )

    population_m3 = sum(value**3 for value in centered) / n
    population_m4 = sum(value**4 for value in centered) / n

    sample_std = sqrt(population_m2 * n / (n - 1))
    skewness = population_m3 / sqrt(population_m2**3)
    excess_kurtosis = population_m4 / population_m2**2 - 3.0

    for name, value in (
        ("mean_return", average),
        ("sample_std", sample_std),
        ("skewness", skewness),
        ("excess_kurtosis", excess_kurtosis),
    ):
        if not isfinite(value):
            raise StatisticalValidationError(f"{name} is not finite")

    return ReturnDistribution(
        observations=n,
        mean_return=average,
        sample_std=sample_std,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        minimum_return=min(values),
        maximum_return=max(values),
    )


def periodic_sharpe_ratio(
    returns: Sequence[float],
    *,
    risk_free_per_period: float = 0.0,
) -> float:
    """Calculate the non-annualized sample Sharpe ratio."""

    risk_free = _finite_float(
        risk_free_per_period,
        "risk_free_per_period",
    )

    values = _normalise_returns(returns)
    excess = [value - risk_free for value in values]
    distribution = return_distribution(excess)
    return distribution.mean_return / distribution.sample_std


def _variance_of_sharpe_estimator(
    *,
    sharpe: float,
    distribution: ReturnDistribution,
) -> float:
    """Approximate variance of the sample Sharpe estimator."""

    numerator = (
        1.0
        - distribution.skewness * sharpe
        + ((distribution.pearson_kurtosis - 1.0) / 4.0) * sharpe**2
    )
    denominator = distribution.observations - 1

    if not isfinite(numerator) or numerator <= EPS:
        raise StatisticalValidationError(
            "Sharpe estimator variance is not positive"
        )

    result = numerator / denominator

    if not isfinite(result) or result <= EPS:
        raise StatisticalValidationError(
            "Sharpe estimator variance is not positive"
        )

    return result


def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    *,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Estimate P(SR > benchmark_sharpe) under the PSR approximation.

    `benchmark_sharpe` must be expressed at the same return frequency as
    `returns`.
    """

    benchmark = _finite_float(
        benchmark_sharpe,
        "benchmark_sharpe",
    )

    values = _normalise_returns(returns)
    distribution = return_distribution(values)
    sharpe = distribution.mean_return / distribution.sample_std
    variance = _variance_of_sharpe_estimator(
        sharpe=sharpe,
        distribution=distribution,
    )
    standard_error = sqrt(variance)
    z_score = (sharpe - benchmark) / standard_error
    probability = NormalDist().cdf(z_score)

    return min(1.0, max(0.0, probability))


def _expected_maximum_sharpe(
    *,
    observed_sharpe: float,
    distribution: ReturnDistribution,
    number_of_trials: int,
    benchmark_sharpe: float,
) -> float:
    """Estimate the expected maximum Sharpe under multiple testing."""

    if isinstance(number_of_trials, bool) or not isinstance(
        number_of_trials,
        int,
    ):
        raise StatisticalValidationError(
            "number_of_trials must be an integer"
        )
    if number_of_trials < 2:
        raise StatisticalValidationError(
            "number_of_trials must be >= 2 for DSR"
        )

    variance = _variance_of_sharpe_estimator(
        sharpe=observed_sharpe,
        distribution=distribution,
    )
    standard_error = sqrt(variance)

    inverse_a = NormalDist().inv_cdf(
        1.0 - 1.0 / number_of_trials
    )
    inverse_b = NormalDist().inv_cdf(
        1.0 - 1.0 / (number_of_trials * e)
    )

    expected_standardised_max = (
        (1.0 - EULER_MASCHERONI) * inverse_a
        + EULER_MASCHERONI * inverse_b
    )

    result = (
        float(benchmark_sharpe)
        + standard_error * expected_standardised_max
    )

    if not isfinite(result):
        raise StatisticalValidationError(
            "expected maximum Sharpe is not finite"
        )

    return result


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    number_of_trials: int,
    benchmark_sharpe: float = 0.0,
) -> tuple[float, float]:
    """
    Return `(deflated_probability, expected_max_sharpe)`.

    The expected maximum Sharpe threshold grows with the number of trials,
    making the resulting probability more conservative after extensive
    strategy/search experimentation.
    """

    benchmark = _finite_float(
        benchmark_sharpe,
        "benchmark_sharpe",
    )
    values = _normalise_returns(returns)
    distribution = return_distribution(values)
    observed = distribution.mean_return / distribution.sample_std

    expected_max = _expected_maximum_sharpe(
        observed_sharpe=observed,
        distribution=distribution,
        number_of_trials=number_of_trials,
        benchmark_sharpe=benchmark,
    )

    variance = _variance_of_sharpe_estimator(
        sharpe=observed,
        distribution=distribution,
    )
    z_score = (observed - expected_max) / sqrt(variance)
    probability = NormalDist().cdf(z_score)

    return min(1.0, max(0.0, probability)), expected_max


def returns_from_equity_curve(
    equity_curve: Sequence[object],
) -> list[float]:
    """
    Convert a Phase 11 equity curve into simple close-to-close returns.

    The function requires strictly increasing timestamps and positive finite
    equity values. It uses only adjacent equity points, so it introduces no
    future information.
    """

    if len(equity_curve) < 2:
        raise StatisticalValidationError(
            "equity_curve requires at least two points"
        )

    timestamps: list[int] = []
    equity: list[float] = []

    for index, point in enumerate(equity_curve):
        try:
            raw_timestamp = getattr(point, "timestamp")
            raw_equity = getattr(point, "equity")
        except AttributeError as exc:
            raise StatisticalValidationError(
                f"invalid equity point at index {index}"
            ) from exc

        if isinstance(raw_timestamp, bool) or not isinstance(
            raw_timestamp,
            int,
        ):
            raise StatisticalValidationError(
                f"timestamp at index {index} must be an integer"
            )

        timestamp = raw_timestamp
        value = _finite_float(
            raw_equity,
            f"equity at index {index}",
        )

        if timestamp <= 0 or value <= 0.0:
            raise StatisticalValidationError(
                f"invalid equity point at index {index}"
            )

        if timestamps and timestamp <= timestamps[-1]:
            raise StatisticalValidationError(
                "equity timestamps must be strictly increasing"
            )

        timestamps.append(timestamp)
        equity.append(value)

    returns = [
        current / previous - 1.0
        for previous, current in zip(
            equity[:-1],
            equity[1:],
        )
    ]

    return _normalise_returns(returns)


def _infer_periods_per_year(
    equity_curve: Sequence[object],
) -> float:
    timestamps = [
        getattr(point, "timestamp")
        for point in equity_curve
    ]
    intervals = [
        (current - previous) / 1000.0
        for previous, current in zip(
            timestamps[:-1],
            timestamps[1:],
        )
    ]
    positive = [
        value
        for value in intervals
        if value > 0.0
    ]

    if not positive:
        raise StatisticalValidationError(
            "equity_curve has no positive time intervals"
        )

    result = SECONDS_PER_YEAR / median(positive)

    if not isfinite(result) or result <= 0.0:
        raise StatisticalValidationError(
            "periods_per_year must be positive and finite"
        )

    return result


def validate_equity_curve(
    equity_curve: Sequence[object],
    *,
    number_of_trials: int,
    benchmark_sharpe: float = 0.0,
) -> StatisticalValidationReport:
    """
    Build a PSR/DSR report directly from a Phase 11 equity curve.

    `number_of_trials` is the number of materially distinct research
    choices that were searched before selecting the reported result. It is
    not the number of trades and should not be guessed from trade count.
    """

    values = returns_from_equity_curve(equity_curve)
    periods_per_year = _infer_periods_per_year(equity_curve)

    distribution = return_distribution(values)
    periodic_sr = (
        distribution.mean_return
        / distribution.sample_std
    )
    annualized_sr = periodic_sr * sqrt(
        periods_per_year
    )

    psr = probabilistic_sharpe_ratio(
        values,
        benchmark_sharpe=benchmark_sharpe,
    )
    dsr, expected_max = deflated_sharpe_ratio(
        values,
        number_of_trials=number_of_trials,
        benchmark_sharpe=benchmark_sharpe,
    )

    return StatisticalValidationReport(
        distribution=distribution,
        periodic_sharpe=periodic_sr,
        annualized_sharpe=annualized_sr,
        benchmark_sharpe=_finite_float(
            benchmark_sharpe,
            "benchmark_sharpe",
        ),
        probabilistic_sharpe=psr,
        number_of_trials=number_of_trials,
        expected_max_sharpe=expected_max,
        deflated_sharpe=dsr,
        periods_per_year=periods_per_year,
    )


__all__ = [
    "ReturnDistribution",
    "StatisticalValidationError",
    "StatisticalValidationReport",
    "deflated_sharpe_ratio",
    "periodic_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "return_distribution",
    "returns_from_equity_curve",
    "validate_equity_curve",
]
