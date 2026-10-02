"""
Tests for EVA-AI-01 Phase 5 Signal Engine.

The tests validate the deterministic, research-oriented contract of the
current R-RSE v1 signal engine.

Important:
- These tests do not claim profitability.
- These tests do not place orders.
- These tests do not connect to Binance.
- The signal engine is treated as a hypothesis generator.
"""

import math

import numpy as np
import pytest

from core.signal_engine import (
    SignalEngineError,
    SignalResult,
    _self_test,
    evaluate,
)


def make_closes(
    rows: int = 60,
    *,
    base: float = 100.0,
    step: float = 0.5,
) -> list[float]:
    """Create deterministic positive close prices."""

    return [
        base + step * i
        for i in range(rows)
    ]


def make_volumes(
    rows: int = 60,
    *,
    base: float = 1000.0,
) -> list[float]:
    """Create deterministic non-negative volume data."""

    return [
        base + float((i % 5) * 25)
        for i in range(rows)
    ]


def evaluate_default(
    closes: list[float] | None = None,
    volumes: list[float] | None = None,
) -> SignalResult:
    """Evaluate a deterministic default test stream."""

    if closes is None:
        closes = make_closes()

    if volumes is None:
        volumes = make_volumes(len(closes))

    return evaluate(
        symbol="TESTUSDT",
        timestamp=1_800_000_000,
        closes=closes,
        volumes=volumes,
    )


def assert_finite_result(
    result: SignalResult,
) -> None:
    """Assert that every numeric result field is finite."""

    numeric_fields = (
        result.timestamp,
        result.direction,
        result.score,
        result.confidence,
        result.classical_z,
        result.robust_z,
        result.volatility,
        result.trend_score,
        result.mean_reversion_score,
        result.volume_score,
        result.regime_factor,
        result.volatility_factor,
    )

    assert all(
        math.isfinite(float(value))
        for value in numeric_fields
    )


def test_evaluate_returns_signal_result():
    result = evaluate_default()

    assert isinstance(result, SignalResult)
    assert result.symbol == "TESTUSDT"
    assert result.timestamp == 1_800_000_000


def test_signal_result_to_dict_is_complete():
    result = evaluate_default()

    payload = result.to_dict()

    expected_keys = {
        "symbol",
        "timestamp",
        "direction",
        "score",
        "confidence",
        "classical_z",
        "robust_z",
        "volatility",
        "trend_score",
        "mean_reversion_score",
        "volume_score",
        "regime_factor",
        "volatility_factor",
        "reasons",
    }

    assert set(payload) == expected_keys

    assert payload["symbol"] == result.symbol
    assert payload["timestamp"] == result.timestamp
    assert payload["direction"] == result.direction
    assert payload["reasons"] == list(result.reasons)


def test_signal_output_is_finite_and_bounded():
    result = evaluate_default()

    assert_finite_result(result)

    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0

    assert -8.0 <= result.classical_z <= 8.0
    assert -8.0 <= result.robust_z <= 8.0

    assert -1.0 <= result.trend_score <= 1.0
    assert -1.0 <= result.mean_reversion_score <= 1.0
    assert -1.0 <= result.volume_score <= 1.0

    assert 0.0 <= result.regime_factor <= 1.0
    assert 0.0 < result.volatility_factor <= 1.0


def test_direction_is_symbolic_and_spot_safe():
    result = evaluate_default()

    assert result.direction in (-1, 0, 1)

    assert not hasattr(result, "quantity")
    assert not hasattr(result, "order_type")
    assert not hasattr(result, "order_id")


def test_flat_market_is_neutral():
    closes = [100.0] * 60
    volumes = [1000.0] * 60

    result = evaluate_default(
        closes,
        volumes,
    )

    assert result.volatility == 0.0
    assert result.trend_score == 0.0
    assert result.mean_reversion_score == 0.0
    assert result.score == 0.0
    assert result.direction == 0

    assert "hysteresis-neutral-zone" in result.reasons


def test_strong_uptrend_generates_positive_hypothesis():
    closes = make_closes(
        rows=60,
        base=100.0,
        step=2.0,
    )

    volumes = make_volumes(60)

    result = evaluate_default(
        closes,
        volumes,
    )

    assert result.trend_score > 0.0
    assert result.score > 0.0
    assert result.direction == 1


def test_strong_downtrend_generates_negative_hypothesis():
    closes = [
        220.0 - 2.0 * i
        for i in range(60)
    ]

    volumes = make_volumes(60)

    result = evaluate_default(
        closes,
        volumes,
    )

    assert result.trend_score < 0.0
    assert result.score < 0.0
    assert result.direction == -1


def test_custom_windows_are_supported():
    closes = make_closes(
        rows=25,
        base=100.0,
        step=0.5,
    )

    volumes = make_volumes(25)

    result = evaluate(
        symbol="TESTUSDT",
        timestamp=1_800_000_000,
        closes=closes,
        volumes=volumes,
        z_window=10,
        trend_window=8,
        volume_window=8,
    )

    assert_finite_result(result)
    assert result.direction in (-1, 0, 1)


def test_insufficient_observations_are_rejected():
    closes = make_closes(39)
    volumes = make_volumes(39)

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )


def test_mismatched_close_and_volume_lengths_are_rejected():
    closes = make_closes(60)
    volumes = make_volumes(59)

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )


def test_non_finite_close_values_are_rejected():
    closes = make_closes(60)
    volumes = make_volumes(60)

    closes[20] = np.nan

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )

    closes[20] = np.inf

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )


def test_non_finite_volume_values_are_rejected():
    closes = make_closes(60)
    volumes = make_volumes(60)

    volumes[20] = np.nan

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )

    volumes[20] = np.inf

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )


def test_non_positive_close_prices_are_rejected():
    closes = make_closes(60)
    volumes = make_volumes(60)

    closes[10] = 0.0

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )

    closes[10] = -1.0

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )


def test_negative_volumes_are_rejected():
    closes = make_closes(60)
    volumes = make_volumes(60)

    volumes[10] = -1.0

    with pytest.raises(SignalEngineError):
        evaluate_default(
            closes,
            volumes,
        )


def test_empty_symbol_is_rejected():
    with pytest.raises(SignalEngineError):
        evaluate(
            symbol="   ",
            timestamp=1_800_000_000,
            closes=make_closes(),
            volumes=make_volumes(),
        )


def test_invalid_timestamp_is_rejected():
    with pytest.raises(SignalEngineError):
        evaluate(
            symbol="TESTUSDT",
            timestamp=0,
            closes=make_closes(),
            volumes=make_volumes(),
        )

    with pytest.raises(SignalEngineError):
        evaluate(
            symbol="TESTUSDT",
            timestamp=-1,
            closes=make_closes(),
            volumes=make_volumes(),
        )


def test_invalid_thresholds_are_rejected():
    closes = make_closes()
    volumes = make_volumes()

    with pytest.raises(SignalEngineError):
        evaluate(
            symbol="TESTUSDT",
            timestamp=1_800_000_000,
            closes=closes,
            volumes=volumes,
            entry_threshold=0.0,
        )

    with pytest.raises(SignalEngineError):
        evaluate(
            symbol="TESTUSDT",
            timestamp=1_800_000_000,
            closes=closes,
            volumes=volumes,
            entry_threshold=1.1,
        )

    with pytest.raises(SignalEngineError):
        evaluate(
            symbol="TESTUSDT",
            timestamp=1_800_000_000,
            closes=closes,
            volumes=volumes,
            entry_threshold=0.5,
            neutral_threshold=0.5,
        )


def test_invalid_ewma_decay_is_rejected():
    closes = make_closes()
    volumes = make_volumes()

    with pytest.raises(SignalEngineError):
        evaluate(
            symbol="TESTUSDT",
            timestamp=1_800_000_000,
            closes=closes,
            volumes=volumes,
            ewma_decay=0.0,
        )

    with pytest.raises(SignalEngineError):
        evaluate(
            symbol="TESTUSDT",
            timestamp=1_800_000_000,
            closes=closes,
            volumes=volumes,
            ewma_decay=1.0,
        )


def test_regime_factor_is_clamped_to_unit_interval():
    closes = make_closes()
    volumes = make_volumes()

    zero_regime = evaluate(
        symbol="TESTUSDT",
        timestamp=1_800_000_000,
        closes=closes,
        volumes=volumes,
        external_regime_factor=-5.0,
    )

    high_regime = evaluate(
        symbol="TESTUSDT",
        timestamp=1_800_000_000,
        closes=closes,
        volumes=volumes,
        external_regime_factor=5.0,
    )

    assert zero_regime.regime_factor == 0.0
    assert zero_regime.score == 0.0
    assert zero_regime.direction == 0

    assert high_regime.regime_factor == 1.0


def test_forecast_volatility_can_only_dampen_conviction():
    closes = make_closes(
        rows=60,
        base=100.0,
        step=2.0,
    )

    volumes = make_volumes(60)

    baseline = evaluate_default(
        closes,
        volumes,
    )

    damped = evaluate(
        symbol="TESTUSDT",
        timestamp=1_800_000_000,
        closes=closes,
        volumes=volumes,
        forecast_volatility=baseline.volatility * 10.0,
    )

    assert (
        damped.volatility_factor
        < baseline.volatility_factor
    )

    assert (
        abs(damped.score)
        <= abs(baseline.score)
    )

    assert_finite_result(damped)


def test_hysteresis_neutralizes_small_scores():
    closes = make_closes(
        rows=60,
        base=100.0,
        step=0.001,
    )

    volumes = [1000.0] * 60

    result = evaluate(
        symbol="TESTUSDT",
        timestamp=1_800_000_000,
        closes=closes,
        volumes=volumes,
        entry_threshold=0.90,
        neutral_threshold=0.80,
    )

    if abs(result.score) < 0.80:
        assert result.direction == 0
        assert "hysteresis-neutral-zone" in result.reasons


def test_output_is_deterministic():
    closes = make_closes(
        rows=60,
        base=100.0,
        step=0.75,
    )

    volumes = make_volumes(60)

    first = evaluate_default(
        closes,
        volumes,
    )

    second = evaluate_default(
        closes,
        volumes,
    )

    assert first == second


def test_reasons_are_immutable_tuple():
    result = evaluate_default()

    assert isinstance(
        result.reasons,
        tuple,
    )

    assert all(
        isinstance(reason, str)
        for reason in result.reasons
    )


def test_internal_self_test_passes():
    _self_test()


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-q",
        ]
    )
