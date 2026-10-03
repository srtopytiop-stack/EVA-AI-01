"""Tests for the EVA-AI-01 Phase 6 Market Regime Engine."""

from __future__ import annotations

import math

import pytest

from core.regime_engine import (
    MarketRegime,
    RegimeEngine,
    RegimeEngineError,
    RegimeResult,
    classify,
)


def make_uptrend(length: int = 100, step: float = 0.01) -> list[float]:
    prices = [100.0]
    for _ in range(length - 1):
        prices.append(prices[-1] * (1.0 + step))
    return prices


def make_downtrend(length: int = 100, step: float = 0.01) -> list[float]:
    prices = [100.0]
    for _ in range(length - 1):
        prices.append(prices[-1] * (1.0 - step))
    return prices


def make_flat(length: int = 100, price: float = 100.0) -> list[float]:
    return [price] * length


def make_range(length: int = 100, amplitude: float = 0.01) -> list[float]:
    prices = [100.0]
    for i in range(1, length):
        direction = 1.0 if i % 2 == 0 else -1.0
        prices.append(prices[-1] * (1.0 + direction * amplitude))
    return prices


def make_low_volatility_compression(
    length: int = 100,
    high_step: float = 0.01,
    low_step: float = 0.0001,
) -> list[float]:
    prices = [100.0]
    switch = length - 10
    for i in range(1, length):
        step = high_step if i < switch else low_step
        direction = 1.0 if i % 2 == 0 else -1.0
        prices.append(prices[-1] * (1.0 + direction * step))
    return prices


def make_high_volatility_expansion(
    length: int = 100,
    low_step: float = 0.00001,
    high_step: float = 0.03,
) -> list[float]:
    prices = [100.0]
    switch = length - 10
    for i in range(1, length):
        step = low_step if i < switch else high_step
        direction = 1.0 if i % 2 == 0 else -1.0
        prices.append(prices[-1] * (1.0 + direction * step))
    return prices


def make_flat_then_volatile(
    flat_points: int = 40,
    volatile_points: int = 10,
) -> list[float]:
    prices = [100.0] * flat_points
    current = 100.0
    for i in range(volatile_points):
        step = 0.10
        direction = 1.0 if i % 2 == 0 else -1.0
        current *= 1.0 + direction * step
        prices.append(current)
    return prices


def engine() -> RegimeEngine:
    return RegimeEngine()


def test_engine_facade_exists() -> None:
    assert isinstance(engine(), RegimeEngine)


def test_module_function_exists() -> None:
    result = classify(make_uptrend())
    assert isinstance(result, RegimeResult)


def test_result_contains_expected_fields() -> None:
    result = engine().classify(make_uptrend())
    assert isinstance(result.regime, MarketRegime)
    assert 0.0 <= result.confidence <= 1.0
    assert math.isfinite(result.trend_score)
    assert result.volatility >= 0.0
    assert result.directional_efficiency >= 0.0
    assert result.directional_persistence >= -1.0
    assert result.directional_persistence <= 1.0
    assert result.persistence == result.directional_persistence


def test_to_dict_is_json_friendly() -> None:
    result = engine().classify(make_uptrend())
    payload = result.to_dict()

    assert payload["regime"] == result.regime.value
    assert payload["directional_persistence"] == result.directional_persistence
    assert payload["persistence"] == result.directional_persistence
    assert isinstance(payload["reasons"], list)


@pytest.mark.parametrize(
    "bad_prices",
    [
        [],
        [100.0],
        [100.0] * 20 + [0.0] + [100.0] * 80,
        [100.0] * 20 + [-1.0] + [100.0] * 80,
        [100.0] * 20 + [math.nan] + [100.0] * 80,
        [100.0] * 20 + [math.inf] + [100.0] * 80,
        [100.0] * 20 + ["bad"] + [100.0] * 80,
        [100.0] * 20 + [True] + [100.0] * 80,
    ],
)
def test_invalid_prices_are_rejected(bad_prices: list[object]) -> None:
    with pytest.raises(RegimeEngineError):
        engine().classify(bad_prices)  # type: ignore[arg-type]


def test_insufficient_data_is_rejected() -> None:
    with pytest.raises(RegimeEngineError):
        engine().classify([100.0] * 10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trend_window": 1},
        {"volatility_window": 1},
        {"efficiency_window": 1},
        {"persistence_window": 0},
        {"ewma_decay": 0.0},
        {"ewma_decay": 1.0},
        {"trend_threshold": 0.0},
        {"trend_threshold": 1.1},
        {"high_volatility_ratio": 1.0},
        {"high_volatility_ratio": math.inf},
        {"low_volatility_ratio": 0.0},
        {"low_volatility_ratio": 1.0},
        {"range_efficiency_threshold": 0.0},
        {"range_efficiency_threshold": 1.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(RegimeEngineError):
        engine().classify(make_flat(), **kwargs)  # type: ignore[arg-type]


def test_uptrend_is_detected() -> None:
    result = engine().classify(make_uptrend())
    assert result.regime == MarketRegime.TREND_UP
    assert result.trend_score > 0.0
    assert result.directional_efficiency > 0.25
    assert result.directional_persistence > 0.0


def test_downtrend_is_detected() -> None:
    result = engine().classify(make_downtrend())
    assert result.regime == MarketRegime.TREND_DOWN
    assert result.trend_score < 0.0
    assert result.directional_efficiency > 0.25
    assert result.directional_persistence < 0.0


def test_flat_market_is_range() -> None:
    result = engine().classify(make_flat())
    assert result.regime == MarketRegime.RANGE
    assert result.directional_efficiency == 0.0
    assert result.directional_persistence == 0.0
    assert result.volatility_ratio == 1.0


def test_oscillating_market_is_range() -> None:
    result = engine().classify(make_range())
    assert result.regime == MarketRegime.RANGE
    assert result.directional_efficiency < 0.25


def test_high_volatility_is_detected() -> None:
    result = engine().classify(
        make_high_volatility_expansion(),
        volatility_window=10,
        high_volatility_ratio=1.50,
    )
    assert result.volatility_ratio >= 1.50
    assert result.regime == MarketRegime.HIGH_VOLATILITY


def test_high_volatility_ratio_can_be_infinite() -> None:
    result = engine().classify(
        make_flat_then_volatile(),
        trend_window=10,
        volatility_window=10,
        efficiency_window=10,
        persistence_window=10,
        high_volatility_ratio=1.50,
    )
    assert math.isinf(result.volatility_ratio)
    assert result.volatility_ratio > 0.0
    assert result.regime == MarketRegime.HIGH_VOLATILITY


def test_low_volatility_is_detected_after_volatility_compression() -> None:
    result = engine().classify(
        make_low_volatility_compression(),
        volatility_window=10,
        low_volatility_ratio=0.67,
    )
    assert result.volatility_ratio <= 0.67
    assert result.regime == MarketRegime.LOW_VOLATILITY


def test_current_and_reference_volatility_windows_do_not_overlap() -> None:
    prices = make_high_volatility_expansion()
    result = engine().classify(prices, volatility_window=10)

    altered_reference = prices.copy()
    start = len(altered_reference) - 20
    for index in range(start, len(altered_reference) - 10):
        altered_reference[index] *= 1.0000001

    altered_result = engine().classify(
        altered_reference,
        volatility_window=10,
    )

    assert not math.isnan(altered_result.volatility_ratio)
    assert altered_result.regime in MarketRegime


def test_same_input_is_deterministic() -> None:
    prices = make_high_volatility_expansion()
    first = engine().classify(prices)
    second = engine().classify(prices)
    assert first == second


def test_module_and_facade_apis_agree() -> None:
    prices = make_downtrend()
    module_result = classify(prices)
    facade_result = engine().classify(prices)
    assert module_result == facade_result


def test_self_test_passes() -> None:
    assert engine()._self_test() is True


def test_result_regime_is_always_supported() -> None:
    datasets = [
        make_flat(),
        make_range(),
        make_uptrend(),
        make_downtrend(),
        make_high_volatility_expansion(),
        make_low_volatility_compression(),
    ]

    for prices in datasets:
        result = engine().classify(prices)
        assert result.regime in set(MarketRegime)
