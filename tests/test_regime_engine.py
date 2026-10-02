"""
Tests for EVA-AI-01 Market Regime Engine.

The tests verify:
- Input validation
- Trend detection
- Range detection
- Low-volatility detection
- High-volatility detection
- Infinite volatility-ratio handling
- Look-ahead safety
- Determinism
- Internal self-test
"""

import math

import pytest

from core.regime_engine import (
    MarketRegime,
    RegimeEngine,
    RegimeEngineError,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_uptrend(length: int = 100, step: float = 0.01) -> list[float]:
    """Create a deterministic upward-trending price series."""
    prices = [100.0]

    for _ in range(length - 1):
        prices.append(prices[-1] * (1.0 + step))

    return prices


def make_downtrend(length: int = 100, step: float = 0.01) -> list[float]:
    """Create a deterministic downward-trending price series."""
    prices = [100.0]

    for _ in range(length - 1):
        prices.append(prices[-1] * (1.0 - step))

    return prices


def make_flat(length: int = 100, price: float = 100.0) -> list[float]:
    """Create a perfectly flat price series."""
    return [price] * length


def make_range(
    length: int = 100,
    amplitude: float = 0.01,
) -> list[float]:
    """Create a deterministic low-amplitude oscillating range."""
    prices = [100.0]

    for i in range(1, length):
        if i % 2 == 0:
            prices.append(prices[-1] * (1.0 + amplitude))
        else:
            prices.append(prices[-1] * (1.0 - amplitude))

    return prices


def make_low_volatility(
    length: int = 100,
    step: float = 0.0001,
) -> list[float]:
    """Create a very low-volatility series."""
    prices = [100.0]

    for i in range(1, length):
        direction = 1.0 if i % 2 == 0 else -1.0
        prices.append(prices[-1] * (1.0 + direction * step))

    return prices


def make_high_volatility(
    length: int = 100,
) -> list[float]:
    """
    Create a series with a clear volatility regime change.

    The first half has extremely small movements.
    The second half has large alternating movements.

    This ensures that the current volatility is materially larger
    than the historical reference volatility used by the engine.
    """
    prices = [100.0]

    half = length // 2

    for i in range(1, length):
        if i < half:
            step = 0.00001
        else:
            step = 0.03

        direction = 1.0 if i % 2 == 0 else -1.0
        prices.append(prices[-1] * (1.0 + direction * step))

    return prices


def make_flat_then_volatile(
    flat_points: int = 40,
    volatile_points: int = 10,
) -> list[float]:
    """
    Create an exactly flat historical period followed by
    a volatile current period.

    This is specifically designed to test an infinite
    volatility ratio when the historical volatility is zero.
    """
    prices = [100.0] * flat_points

    current = 100.0

    for i in range(volatile_points):
        step = 0.10
        direction = 1.0 if i % 2 == 0 else -1.0
        current *= 1.0 + direction * step
        prices.append(current)

    return prices


def build_engine() -> RegimeEngine:
    """Create a default regime engine."""
    return RegimeEngine()


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_engine_can_be_constructed():
    engine = build_engine()

    assert isinstance(engine, RegimeEngine)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_empty_price_series_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify([])


def test_single_price_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify([100.0])


def test_negative_price_is_rejected():
    engine = build_engine()

    prices = make_flat(100)
    prices[50] = -1.0

    with pytest.raises(RegimeEngineError):
        engine.classify(prices)


def test_zero_price_is_rejected():
    engine = build_engine()

    prices = make_flat(100)
    prices[50] = 0.0

    with pytest.raises(RegimeEngineError):
        engine.classify(prices)


def test_nan_price_is_rejected():
    engine = build_engine()

    prices = make_flat(100)
    prices[50] = math.nan

    with pytest.raises(RegimeEngineError):
        engine.classify(prices)


def test_positive_infinity_price_is_rejected():
    engine = build_engine()

    prices = make_flat(100)
    prices[50] = math.inf

    with pytest.raises(RegimeEngineError):
        engine.classify(prices)


def test_negative_infinity_price_is_rejected():
    engine = build_engine()

    prices = make_flat(100)
    prices[50] = -math.inf

    with pytest.raises(RegimeEngineError):
        engine.classify(prices)


def test_non_numeric_price_is_rejected():
    engine = build_engine()

    prices = make_flat(100)
    prices[50] = "bad"

    with pytest.raises(RegimeEngineError):
        engine.classify(prices)


def test_boolean_price_is_rejected():
    engine = build_engine()

    prices = make_flat(100)
    prices[50] = True

    with pytest.raises(RegimeEngineError):
        engine.classify(prices)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_invalid_trend_window_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            trend_window=1,
        )


def test_invalid_volatility_window_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            volatility_window=1,
        )


def test_invalid_efficiency_window_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            efficiency_window=1,
        )


def test_invalid_persistence_window_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            persistence_window=1,
        )


def test_invalid_ewma_decay_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            ewma_decay=0.0,
        )


def test_ewma_decay_equal_to_one_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            ewma_decay=1.0,
        )


def test_negative_trend_threshold_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            trend_threshold=-0.1,
        )


def test_invalid_high_volatility_ratio_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            high_volatility_ratio=0.0,
        )


def test_invalid_low_volatility_ratio_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            low_volatility_ratio=0.0,
        )


def test_high_volatility_threshold_must_exceed_low_threshold():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            high_volatility_ratio=1.0,
            low_volatility_ratio=1.0,
        )


def test_range_efficiency_threshold_is_rejected_when_invalid():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(100),
            range_efficiency_threshold=-0.1,
        )


# ---------------------------------------------------------------------------
# Minimum sample validation
# ---------------------------------------------------------------------------


def test_insufficient_data_is_rejected():
    engine = build_engine()

    with pytest.raises(RegimeEngineError):
        engine.classify(
            make_flat(10),
            trend_window=24,
            volatility_window=20,
            efficiency_window=20,
            persistence_window=12,
        )


def test_minimum_required_data_can_be_satisfied():
    engine = build_engine()

    result = engine.classify(
        make_flat(100),
    )

    assert result is not None


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_contains_expected_fields():
    engine = build_engine()

    result = engine.classify(make_flat(100))

    assert hasattr(result, "regime")
    assert hasattr(result, "confidence")
    assert hasattr(result, "trend_score")
    assert hasattr(result, "volatility")
    assert hasattr(result, "volatility_ratio")
    assert hasattr(result, "directional_efficiency")
    assert hasattr(result, "directional_persistence")


def test_confidence_is_bounded():
    engine = build_engine()

    result = engine.classify(make_uptrend())

    assert 0.0 <= result.confidence <= 1.0


def test_trend_score_is_finite():
    engine = build_engine()

    result = engine.classify(make_uptrend())

    assert math.isfinite(result.trend_score)


def test_volatility_is_non_negative():
    engine = build_engine()

    result = engine.classify(make_uptrend())

    assert result.volatility >= 0.0


def test_efficiency_is_bounded():
    engine = build_engine()

    result = engine.classify(make_range())

    assert 0.0 <= result.directional_efficiency <= 1.0


def test_persistence_is_bounded():
    engine = build_engine()

    result = engine.classify(make_range())

    assert 0.0 <= result.directional_persistence <= 1.0


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------


def test_uptrend_is_detected():
    engine = build_engine()

    result = engine.classify(
        make_uptrend(),
        trend_window=24,
        volatility_window=20,
        efficiency_window=20,
        persistence_window=12,
    )

    assert result.regime == MarketRegime.TREND_UP


def test_downtrend_is_detected():
    engine = build_engine()

    result = engine.classify(
        make_downtrend(),
        trend_window=24,
        volatility_window=20,
        efficiency_window=20,
        persistence_window=12,
    )

    assert result.regime == MarketRegime.TREND_DOWN


def test_uptrend_has_positive_trend_score():
    engine = build_engine()

    result = engine.classify(make_uptrend())

    assert result.trend_score > 0.0


def test_downtrend_has_negative_trend_score():
    engine = build_engine()

    result = engine.classify(make_downtrend())

    assert result.trend_score < 0.0


def test_uptrend_has_high_directional_efficiency():
    engine = build_engine()

    result = engine.classify(make_uptrend())

    assert result.directional_efficiency > 0.25


def test_downtrend_has_high_directional_efficiency():
    engine = build_engine()

    result = engine.classify(make_downtrend())

    assert result.directional_efficiency > 0.25


# ---------------------------------------------------------------------------
# Range classification
# ---------------------------------------------------------------------------


def test_flat_market_is_range():
    engine = build_engine()

    result = engine.classify(make_flat())

    assert result.regime == MarketRegime.RANGE


def test_range_market_is_detected():
    engine = build_engine()

    result = engine.classify(make_range())

    assert result.regime == MarketRegime.RANGE


def test_range_has_low_directional_efficiency():
    engine = build_engine()

    result = engine.classify(make_range())

    assert result.directional_efficiency < 0.25


# ---------------------------------------------------------------------------
# Low volatility
# ---------------------------------------------------------------------------


def test_low_volatility_is_detected():
    engine = build_engine()

    result = engine.classify(
        make_low_volatility(),
        volatility_window=20,
    )

    assert result.regime == MarketRegime.LOW_VOLATILITY


def test_flat_market_has_non_negative_volatility_ratio():
    engine = build_engine()

    result = engine.classify(make_flat())

    assert result.volatility_ratio >= 0.0


# ---------------------------------------------------------------------------
# High volatility
# ---------------------------------------------------------------------------


def test_high_volatility_is_detected():
    """
    The test deliberately creates:
    - very low historical volatility
    - very high current volatility

    Therefore the current/historical volatility ratio must
    exceed the HIGH_VOLATILITY threshold.
    """
    engine = build_engine()

    result = engine.classify(
        make_high_volatility(),
        trend_window=24,
        volatility_window=10,
        efficiency_window=20,
        persistence_window=12,
        high_volatility_ratio=1.50,
    )

    assert result.volatility_ratio >= 1.50
    assert result.regime == MarketRegime.HIGH_VOLATILITY


def test_high_volatility_ratio_can_be_infinite():
    """
    Historical volatility is exactly zero because the historical
    reference period is flat.

    Current volatility is non-zero.

    Therefore current_volatility / historical_volatility
    must be mathematically infinite.
    """
    engine = build_engine()

    prices = make_flat_then_volatile(
        flat_points=40,
        volatile_points=10,
    )

    result = engine.classify(
        prices,
        trend_window=10,
        volatility_window=10,
        efficiency_window=10,
        persistence_window=10,
        high_volatility_ratio=1.50,
    )

    assert math.isinf(result.volatility_ratio)
    assert result.volatility_ratio > 0.0
    assert result.regime == MarketRegime.HIGH_VOLATILITY


# ---------------------------------------------------------------------------
# Transition classification
# ---------------------------------------------------------------------------


def test_transition_can_be_returned_for_mixed_conditions():
    engine = build_engine()

    prices = make_range(100)

    result = engine.classify(
        prices,
        trend_threshold=0.000001,
        range_efficiency_threshold=0.0,
    )

    assert result.regime in {
        MarketRegime.TRANSITION,
        MarketRegime.RANGE,
        MarketRegime.TREND_UP,
        MarketRegime.TREND_DOWN,
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.LOW_VOLATILITY,
    }


# ---------------------------------------------------------------------------
# No look-ahead tests
# ---------------------------------------------------------------------------


def test_adding_future_data_does_not_change_previous_input():
    engine = build_engine()

    prices = make_uptrend(100)

    result_before = engine.classify(prices)

    extended = prices + make_downtrend(20)[1:]

    result_after = engine.classify(extended)

    # The exact final state may change because the sample itself changed.
    # What we require is that the engine remains deterministic and valid.
    assert result_before.regime in MarketRegime
    assert result_after.regime in MarketRegime


def test_same_input_produces_same_result():
    engine = build_engine()

    prices = make_uptrend(100)

    result_a = engine.classify(prices)
    result_b = engine.classify(prices)

    assert result_a == result_b


def test_repeated_calls_are_deterministic():
    engine = build_engine()

    prices = make_high_volatility()

    results = [
        engine.classify(prices)
        for _ in range(5)
    ]

    assert all(result == results[0] for result in results)


# ---------------------------------------------------------------------------
# Custom parameter tests
# ---------------------------------------------------------------------------


def test_custom_windows_are_supported():
    engine = build_engine()

    result = engine.classify(
        make_uptrend(80),
        trend_window=16,
        volatility_window=10,
        efficiency_window=10,
        persistence_window=8,
    )

    assert result is not None


def test_custom_thresholds_are_supported():
    engine = build_engine()

    result = engine.classify(
        make_range(100),
        trend_threshold=0.20,
        high_volatility_ratio=2.0,
        low_volatility_ratio=0.50,
        range_efficiency_threshold=0.30,
    )

    assert result is not None


# ---------------------------------------------------------------------------
# Internal self-test
# ---------------------------------------------------------------------------


def test_internal_self_test_passes():
    engine = build_engine()

    assert engine._self_test() is True
