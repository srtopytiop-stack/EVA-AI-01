"""Phase 12 statistical-validation tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt

import pytest

from core.statistical_validation import (
    StatisticalValidationError,
    deflated_sharpe_ratio,
    periodic_sharpe_ratio,
    probabilistic_sharpe_ratio,
    return_distribution,
    returns_from_equity_curve,
    validate_equity_curve,
)


@dataclass(frozen=True)
class MockEquityPoint:
    timestamp: int
    equity: float


def sample_returns() -> list[float]:
    return [
        -0.005,
        0.010,
        0.004,
        0.012,
        -0.003,
        0.011,
        0.006,
        0.009,
        0.005,
        0.008,
        0.007,
        0.010,
    ]


def test_distribution_uses_sample_standard_deviation() -> None:
    values = [0.01, 0.03, -0.01, 0.02]

    distribution = return_distribution(values)

    assert distribution.observations == 4
    assert distribution.mean_return == pytest.approx(0.0125)
    assert distribution.sample_std == pytest.approx(0.01707825127659933)
    assert periodic_sharpe_ratio(values) == pytest.approx(
        0.7319250547114,
    )


def test_probabilistic_sharpe_ratio_is_a_probability() -> None:
    values = sample_returns()

    probability_zero = probabilistic_sharpe_ratio(
        values,
        benchmark_sharpe=0.0,
    )
    probability_positive_benchmark = probabilistic_sharpe_ratio(
        values,
        benchmark_sharpe=0.1,
    )

    assert 0.0 <= probability_zero <= 1.0
    assert 0.0 <= probability_positive_benchmark <= 1.0
    assert probability_zero > probability_positive_benchmark


def test_deflated_sharpe_gets_more_conservative_with_more_trials() -> None:
    values = sample_returns()

    dsr_2, expected_max_2 = deflated_sharpe_ratio(
        values,
        number_of_trials=2,
    )
    dsr_100, expected_max_100 = deflated_sharpe_ratio(
        values,
        number_of_trials=100,
    )

    assert expected_max_100 > expected_max_2
    assert dsr_100 < dsr_2
    assert 0.0 <= dsr_2 <= 1.0
    assert 0.0 <= dsr_100 <= 1.0


def test_equity_curve_is_converted_without_future_information() -> None:
    base = 1_800_000_000_000
    curve = [
        MockEquityPoint(
            timestamp=base + index * 60_000,
            equity=1_000.0 + index * (3.0 if index % 2 else 1.0),
        )
        for index in range(12)
    ]

    returns = returns_from_equity_curve(curve)

    assert len(returns) == len(curve) - 1
    assert returns[0] == pytest.approx(0.003)


def test_validate_equity_curve_infers_annualization_and_is_json_safe() -> None:
    base = 1_800_000_000_000
    equity_values = [
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
    ]
    curve = [
        MockEquityPoint(
            timestamp=base + index * 60_000,
            equity=value,
        )
        for index, value in enumerate(equity_values)
    ]

    report = validate_equity_curve(
        curve,
        number_of_trials=25,
    )

    assert report.distribution.observations == len(curve) - 1
    assert report.periods_per_year == pytest.approx(
        (365.25 * 24.0 * 60.0 * 60.0) / 60.0,
    )
    assert report.annualized_sharpe == pytest.approx(
        report.periodic_sharpe * sqrt(report.periods_per_year),
    )
    assert 0.0 <= report.probabilistic_sharpe <= 1.0
    assert 0.0 <= report.deflated_sharpe <= 1.0

    json.dumps(report.to_dict(), allow_nan=False)


def test_zero_variance_returns_are_rejected() -> None:
    with pytest.raises(StatisticalValidationError):
        return_distribution([0.01, 0.01, 0.01, 0.01])


def test_invalid_trial_count_is_rejected() -> None:
    with pytest.raises(StatisticalValidationError):
        deflated_sharpe_ratio(
            sample_returns(),
            number_of_trials=1,
        )

    with pytest.raises(StatisticalValidationError):
        deflated_sharpe_ratio(
            sample_returns(),
            number_of_trials=True,
        )


def test_invalid_equity_curve_is_rejected() -> None:
    base = 1_800_000_000_000

    with pytest.raises(StatisticalValidationError):
        returns_from_equity_curve([])

    with pytest.raises(StatisticalValidationError):
        returns_from_equity_curve(
            [
                MockEquityPoint(base, 1_000.0),
                MockEquityPoint(base, 1_001.0),
            ]
        )

    with pytest.raises(StatisticalValidationError):
        returns_from_equity_curve(
            [
                MockEquityPoint(base, 1_000.0),
                MockEquityPoint(base + 60_000, 0.0),
            ]
        )


def test_nonfinite_inputs_are_rejected() -> None:
    with pytest.raises(StatisticalValidationError):
        probabilistic_sharpe_ratio(
            sample_returns(),
            benchmark_sharpe=float("nan"),
        )

    with pytest.raises(StatisticalValidationError):
        return_distribution(
            [0.01, float("inf"), 0.02, 0.03]
        )
