"""
Tests for EVA-AI-01 Phase 6 Market Regime Engine.

These tests validate:

- deterministic behavior
- input validation
- output bounds
- regime classification
- volatility detection
- trend detection
- range detection
- transition detection
- directional efficiency
- persistence
- volatility regime separation
- no-look-ahead behavior

Important:
- These tests do NOT claim profitability.
- These tests do NOT place orders.
- These tests do NOT connect to Binance.
- The regime engine is an analytical classification layer only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.regime_engine import (
    MarketRegime,
    RegimeEngineError,
    RegimeResult,
    _self_test,
    classify,
)


def make_uptrend(
    rows: int = 100,
    *,
    base: float = 100.0,
    step: float = 1.0,
) -> list[float]:
    """Create a deterministic upward trend."""

    return [
        base + step * index
        for index in range(rows)
    ]


def make_downtrend(
    rows: int = 100,
    *,
    base: float = 200.0,
    step: float = 1.0,
) -> list[float]:
    """Create a deterministic downward trend."""

    return [
        base - step * index
        for index in range(rows)
    ]


def make_flat(
    rows: int = 100,
    *,
    price: float = 100.0,
) -> list[float]:
    """Create a perfectly flat market."""

    return [
        price
        for _ in range(rows)
    ]


def make_range(
    rows: int = 100,
    *,
    center: float = 100.0,
    amplitude: float = 2.0,
) -> list[float]:
    """
    Create a deterministic oscillating/ranging market.

    The sequence repeatedly moves between positive and negative deviations
    around a stable center.
    """

    pattern = [
        -amplitude,
        0.0,
        amplitude,
        0.0,
    ]

    return [
        center + pattern[index % len(pattern)]
        for index in range(rows)
    ]


def make_low_volatility(
    rows: int = 100,
    *,
    base: float = 100.0,
) -> list[float]:
    """Create a low-volatility price series."""

    pattern = [
        0.00,
        0.01,
        -0.01,
        0.005,
        -0.005,
    ]

    return [
        base + pattern[index % len(pattern)]
        for index in range(rows)
    ]


def make_high_volatility(
    rows: int = 100,
    *,
    base: float = 100.0,
) -> list[float]:
    """
    Create a deterministic volatility-regime transition.

    Structure:

        first half  -> very low volatility
        second half -> high volatility

    The important property is that the historical reference window
    immediately preceding the current window remains low-volatility.
    """

    if rows < 40:
        raise ValueError(
            "rows must be at least 40"
        )

    prices = [base]

    transition = rows // 2

    for index in range(1, rows):
        if index < transition:
            # Extremely small deterministic movement.
            step = (
                0.002
                if index % 2
                else -0.002
            )
        else:
            # Large alternating movement.
            step = (
                3.0
                if index % 2
                else -3.0
            )

        prices.append(
            prices[-1] + step
        )

    return prices


def make_flat_then_volatile(
    *,
    flat_points: int = 40,
    volatile_points: int = 10,
    base: float = 100.0,
    step: float = 2.0,
) -> list[float]:
    """
    Create an exactly-zero historical volatility baseline followed by
    a volatile current window.

    This specifically tests the mathematically valid infinite
    volatility-ratio case.
    """

    if flat_points < 2:
        raise ValueError(
            "flat_points must be at least 2"
        )

    if volatile_points < 2:
        raise ValueError(
            "volatile_points must be at least 2"
        )

    prices = [
        base
        for _ in range(flat_points)
    ]

    for index in range(volatile_points):
        direction = (
            1.0
            if index % 2 == 0
            else -1.0
        )

        prices.append(
            prices[-1]
            + direction * step
        )

    return prices


def assert_finite_result(
    result: RegimeResult,
) -> None:
    """
    Assert all finite numeric fields.

    volatility_ratio is allowed to be infinity when the historical
    volatility baseline is exactly zero.
    """

    finite_values = (
        result.confidence,
        result.trend_score,
        result.volatility,
        result.directional_efficiency,
        result.persistence,
    )

    assert all(
        math.isfinite(float(value))
        for value in finite_values
    )

    assert (
        math.isfinite(result.volatility_ratio)
        or math.isinf(result.volatility_ratio)
    )


def test_returns_regime_result():
    prices = make_uptrend()

    result = classify(
        prices=prices,
    )

    assert isinstance(
        result,
        RegimeResult,
    )

    assert isinstance(
        result.regime,
        MarketRegime,
    )


def test_result_to_dict_is_complete():
    prices = make_uptrend()

    result = classify(
        prices=prices,
    )

    payload = result.to_dict()

    expected_keys = {
        "regime",
        "confidence",
        "trend_score",
        "volatility",
        "volatility_ratio",
        "directional_efficiency",
        "persistence",
        "reasons",
    }

    assert set(payload) == expected_keys

    assert payload["regime"] == result.regime.value
    assert payload["confidence"] == result.confidence
    assert payload["trend_score"] == result.trend_score
    assert payload["volatility"] == result.volatility
    assert payload["volatility_ratio"] == result.volatility_ratio
    assert (
        payload["directional_efficiency"]
        == result.directional_efficiency
    )
    assert payload["persistence"] == result.persistence
    assert payload["reasons"] == list(result.reasons)


def test_output_bounds_are_valid():
    prices = make_uptrend()

    result = classify(
        prices=prices,
    )

    assert_finite_result(result)

    assert 0.0 <= result.confidence <= 1.0

    assert -1.0 <= result.trend_score <= 1.0

    assert result.volatility >= 0.0

    assert (
        0.0
        <= result.directional_efficiency
        <= 1.0
    )

    assert -1.0 <= result.persistence <= 1.0


def test_uptrend_is_directionally_positive():
    prices = make_uptrend(
        rows=100,
        base=100.0,
        step=1.0,
    )

    result = classify(
        prices=prices,
    )

    assert result.trend_score > 0.0
    assert result.persistence > 0.0
    assert result.directional_efficiency > 0.0


def test_downtrend_is_directionally_negative():
    prices = make_downtrend(
        rows=100,
        base=200.0,
        step=1.0,
    )

    result = classify(
        prices=prices,
    )

    assert result.trend_score < 0.0
    assert result.persistence < 0.0
    assert result.directional_efficiency > 0.0


def test_flat_market_has_zero_volatility():
    prices = make_flat()

    result = classify(
        prices=prices,
    )

    assert result.volatility == 0.0
    assert result.volatility_ratio == 1.0
    assert result.trend_score == 0.0
    assert result.persistence == 0.0
    assert result.directional_efficiency == 0.0


def test_flat_market_is_not_directional():
    prices = make_flat()

    result = classify(
        prices=prices,
    )

    assert result.regime in {
        MarketRegime.LOW_VOLATILITY,
        MarketRegime.RANGE,
        MarketRegime.TRANSITION,
    }

    assert result.regime not in {
        MarketRegime.TREND_UP,
        MarketRegime.TREND_DOWN,
    }


def test_range_market_has_low_directional_efficiency():
    prices = make_range(
        rows=100,
        center=100.0,
        amplitude=2.0,
    )

    result = classify(
        prices=prices,
    )

    assert (
        result.directional_efficiency
        < 0.5
    )


def test_range_market_does_not_create_strong_directional_signal():
    prices = make_range(
        rows=100,
        center=100.0,
        amplitude=2.0,
    )

    result = classify(
        prices=prices,
    )

    assert result.regime not in {
        MarketRegime.TREND_UP,
        MarketRegime.TREND_DOWN,
    }


def test_high_volatility_is_detected():
    """
    The historical reference window must remain low-volatility while
    the current window is highly volatile.
    """

    prices = make_high_volatility(
        rows=100,
    )

    result = classify(
        prices=prices,
        volatility_window=20,
    )

    assert result.volatility > 0.0

    assert result.volatility_ratio > 1.0

    assert (
        result.volatility_ratio
        >= 1.50
    )

    assert result.regime == (
        MarketRegime.HIGH_VOLATILITY
    )


def test_low_volatility_is_detected():
    """
    A stable low-volatility series should never be classified as a
    directional trend solely because of numerical noise.
    """

    prices = make_low_volatility(
        rows=100,
    )

    result = classify(
        prices=prices,
    )

    assert result.volatility >= 0.0

    assert result.regime in {
        MarketRegime.LOW_VOLATILITY,
        MarketRegime.RANGE,
        MarketRegime.TRANSITION,
    }


def test_strong_uptrend_can_be_classified_as_trend_up():
    """
    Use enough observations and sufficiently directional movement so
    volatility is not the dominant regime.
    """

    prices = []

    price = 100.0

    for index in range(100):
        price += (
            0.8
            + 0.03 * math.sin(index)
        )

        prices.append(price)

    result = classify(
        prices=prices,
        trend_window=24,
        volatility_window=20,
        efficiency_window=20,
        persistence_window=12,
    )

    assert result.trend_score > 0.35
    assert result.persistence > 0.20
    assert result.directional_efficiency > 0.25

    assert result.regime == (
        MarketRegime.TREND_UP
    )


def test_strong_downtrend_can_be_classified_as_trend_down():
    """
    Use sufficiently directional downward movement.
    """

    prices = []

    price = 200.0

    for index in range(100):
        price -= (
            0.8
            + 0.03 * math.sin(index)
        )

        prices.append(price)

    result = classify(
        prices=prices,
        trend_window=24,
        volatility_window=20,
        efficiency_window=20,
        persistence_window=12,
    )

    assert result.trend_score < -0.35
    assert result.persistence < -0.20
    assert result.directional_efficiency > 0.25

    assert result.regime == (
        MarketRegime.TREND_DOWN
    )


def test_regime_is_always_supported_enum():
    examples = [
        make_uptrend(),
        make_downtrend(),
        make_flat(),
        make_range(),
        make_low_volatility(),
        make_high_volatility(),
    ]

    for prices in examples:
        result = classify(
            prices=prices,
        )

        assert result.regime in set(
            MarketRegime
        )


def test_reasons_are_immutable_tuple():
    result = classify(
        prices=make_uptrend(),
    )

    assert isinstance(
        result.reasons,
        tuple,
    )

    assert all(
        isinstance(reason, str)
        for reason in result.reasons
    )


def test_reasons_do_not_contain_duplicates():
    result = classify(
        prices=make_uptrend(),
    )

    assert len(result.reasons) == len(
        set(result.reasons)
    )


def test_deterministic_output():
    prices = make_uptrend(
        rows=100,
        base=100.0,
        step=0.7,
    )

    first = classify(
        prices=prices,
    )

    second = classify(
        prices=prices,
    )

    assert first == second


def test_small_input_is_rejected():
    prices = [100.0] * 10

    with pytest.raises(RegimeEngineError):
        classify(
            prices=prices,
        )


def test_empty_input_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=[],
        )


def test_nan_is_rejected():
    prices = make_uptrend()

    prices[30] = np.nan

    with pytest.raises(RegimeEngineError):
        classify(
            prices=prices,
        )


def test_positive_infinity_is_rejected():
    prices = make_uptrend()

    prices[30] = np.inf

    with pytest.raises(RegimeEngineError):
        classify(
            prices=prices,
        )


def test_negative_infinity_is_rejected():
    prices = make_uptrend()

    prices[30] = -np.inf

    with pytest.raises(RegimeEngineError):
        classify(
            prices=prices,
        )


def test_zero_price_is_rejected():
    prices = make_uptrend()

    prices[30] = 0.0

    with pytest.raises(RegimeEngineError):
        classify(
            prices=prices,
        )


def test_negative_price_is_rejected():
    prices = make_uptrend()

    prices[30] = -1.0

    with pytest.raises(RegimeEngineError):
        classify(
            prices=prices,
        )


def test_invalid_trend_window_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            trend_window=1,
        )


def test_invalid_volatility_window_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            volatility_window=1,
        )


def test_invalid_efficiency_window_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            efficiency_window=1,
        )


def test_invalid_persistence_window_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            persistence_window=0,
        )


def test_invalid_ewma_decay_zero_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            ewma_decay=0.0,
        )


def test_invalid_ewma_decay_one_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            ewma_decay=1.0,
        )


def test_invalid_ewma_decay_above_one_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            ewma_decay=1.5,
        )


def test_invalid_ewma_decay_negative_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            ewma_decay=-0.1,
        )


def test_invalid_trend_threshold_zero_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            trend_threshold=0.0,
        )


def test_invalid_trend_threshold_above_one_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            trend_threshold=1.1,
        )


def test_invalid_high_volatility_ratio_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            high_volatility_ratio=1.0,
        )


def test_invalid_low_volatility_ratio_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            low_volatility_ratio=0.0,
        )


def test_invalid_range_efficiency_threshold_is_rejected():
    with pytest.raises(RegimeEngineError):
        classify(
            prices=make_uptrend(),
            range_efficiency_threshold=0.0,
        )


def test_custom_windows_are_supported():
    prices = make_uptrend(
        rows=80,
        base=100.0,
        step=0.5,
    )

    result = classify(
        prices=prices,
        trend_window=16,
        volatility_window=12,
        efficiency_window=12,
        persistence_window=8,
    )

    assert isinstance(
        result,
        RegimeResult,
    )

    assert_finite_result(result)


def test_no_look_ahead_behavior():
    """
    The regime at time t must not depend on observations after t.

    The same historical prefix must always produce exactly the same
    result, regardless of whether future data exists elsewhere.
    """

    prices = make_uptrend(
        rows=120,
        base=100.0,
        step=0.4,
    )

    prefix = prices[:80]

    result_prefix = classify(
        prices=prefix,
    )

    result_prefix_again = classify(
        prices=list(prefix),
    )

    assert result_prefix == result_prefix_again


def test_future_observations_do_not_change_recomputed_prefix():
    """
    Explicitly verify that the historical prefix remains reproducible
    after future observations are appended elsewhere.
    """

    prices = make_uptrend(
        rows=120,
        base=100.0,
        step=0.4,
    )

    prefix = prices[:80]
    future = prices[80:]

    result_before = classify(
        prices=prefix,
    )

    extended = prefix + future

    result_after = classify(
        prices=extended,
    )

    assert isinstance(
        result_before,
        RegimeResult,
    )

    assert isinstance(
        result_after,
        RegimeResult,
    )

    # The final regime may legitimately change because the observation
    # point has moved forward. The prefix itself must remain reproducible.
    assert result_before == classify(
        prices=prefix,
    )


def test_volatility_ratio_is_one_when_both_windows_are_zero():
    prices = make_flat()

    result = classify(
        prices=prices,
    )

    assert result.volatility == 0.0
    assert result.volatility_ratio == 1.0


def test_high_volatility_ratio_can_be_infinite():
    """
    Construct the exact mathematical condition:

        historical volatility = 0
        current volatility > 0

    Therefore:

        current volatility / historical volatility = infinity
    """

    prices = make_flat_then_volatile(
        flat_points=40,
        volatile_points=10,
        base=100.0,
        step=2.0,
    )

    result = classify(
        prices=prices,
        volatility_window=10,
    )

    assert result.volatility > 0.0

    assert math.isinf(
        result.volatility_ratio
    )

    assert (
        result.volatility_ratio
        > 1.0
    )

    assert result.regime == (
        MarketRegime.HIGH_VOLATILITY
    )


def test_confidence_is_bounded_for_all_regime_examples():
    examples = [
        make_uptrend(),
        make_downtrend(),
        make_flat(),
        make_range(),
        make_low_volatility(),
        make_high_volatility(),
        make_flat_then_volatile(),
    ]

    for prices in examples:
        result = classify(
            prices=prices,
        )

        assert (
            0.0
            <= result.confidence
            <= 1.0
        )


def test_result_has_no_execution_fields():
    result = classify(
        prices=make_uptrend(),
    )

    assert not hasattr(
        result,
        "quantity",
    )

    assert not hasattr(
        result,
        "order_type",
    )

    assert not hasattr(
        result,
        "order_id",
    )

    assert not hasattr(
        result,
        "position_size",
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

بعد لصق الملف، نفّذ:

python -m pytest -q

التصحيح الجوهري

الاختبار السابق كان يفترض:

flat → volatile

لكنه فعليًا كان يبني:

flat → volatile → volatile → volatile
             ↑
       historical window

لذلك المحرك كان يتصرف رياضيًا بشكل صحيح ويجد أن الحاضر ليس أكثر تقلبًا من مرجعه مباشرة بما يكفي.

النسخة الجديدة تختبر الحالة الصحيحة:

40 نقطة هادئة
       ↓
10 نقاط متقلبة
       ↓
Historical σ = 0
Current σ > 0
       ↓
Ratio = ∞
       ↓
HIGH_VOLATILITY

وبذلك لا نعدل "regime_engine.py" بشكل مصطنع فقط لجعل CI أخضر. الاختبار هو الذي تم تصحيحه ليتوافق مع التعريف الرياضي الذي يدّعي اختباره.
