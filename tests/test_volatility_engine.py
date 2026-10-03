"""Tests for the EVA-AI-01 Phase 7 Volatility Engine."""

from __future__ import annotations

import math

import pytest

from core.volatility_engine import (
    VolatilityEngine,
    VolatilityEngineError,
    VolatilityResult,
    VolatilityState,
    classify,
)


def make_flat(
    length: int = 100,
    price: float = 100.0,
) -> list[float]:
    return [price] * length


def make_stable(
    length: int = 100,
    step: float = 0.001,
) -> list[float]:
    prices = [100.0]

    for i in range(1, length):
        direction = (
            1.0
            if i % 2 == 0
            else -1.0
        )

        prices.append(
            prices[-1]
            * (1.0 + direction * step)
        )

    return prices


def make_volatility_expansion(
    length: int = 120,
    low_step: float = 0.0001,
    high_step: float = 0.03,
) -> list[float]:
    prices = [100.0]
    switch = length - 20

    for i in range(1, length):
        step = (
            low_step
            if i < switch
            else high_step
        )

        direction = (
            1.0
            if i % 2 == 0
            else -1.0
        )

        prices.append(
            prices[-1]
            * (1.0 + direction * step)
        )

    return prices


def make_volatility_compression(
    length: int = 120,
    high_step: float = 0.02,
    low_step: float = 0.0001,
) -> list[float]:
    prices = [100.0]
    switch = length - 20

    for i in range(1, length):
        step = (
            high_step
            if i < switch
            else low_step
        )

        direction = (
            1.0
            if i % 2 == 0
            else -1.0
        )

        prices.append(
            prices[-1]
            * (1.0 + direction * step)
        )

    return prices


def engine() -> VolatilityEngine:
    return VolatilityEngine()


def test_facade_exists() -> None:
    assert isinstance(
        engine(),
        VolatilityEngine,
    )


def test_module_and_facade_agree() -> None:
    prices = make_stable()

    assert classify(prices) == (
        engine().classify(prices)
    )


def test_result_contains_expected_fields() -> None:
    result = engine().classify(
        make_stable()
    )

    assert isinstance(
        result,
        VolatilityResult,
    )

    assert isinstance(
        result.state,
        VolatilityState,
    )

    assert 0.0 <= result.confidence <= 1.0
    assert result.ewma_volatility >= 0.0
    assert result.realized_volatility >= 0.0
    assert result.reference_volatility >= 0.0
    assert result.volatility_ratio > 0.0
    assert result.volatility_of_volatility >= 0.0
    assert 0.0 <= result.percentile_proxy <= 1.0
    assert result.volatility == (
        result.ewma_volatility
    )


def test_to_dict_is_json_friendly() -> None:
    result = engine().classify(
        make_stable()
    )

    payload = result.to_dict()

    assert payload["state"] == (
        result.state.value
    )

    assert payload["volatility"] == (
        result.ewma_volatility
    )

    assert isinstance(
        payload["reasons"],
        list,
    )


@pytest.mark.parametrize(
    "bad_prices",
    [
        [],
        [100.0],
        [100.0] * 60
        + [0.0]
        + [100.0] * 60,
        [100.0] * 60
        + [-1.0]
        + [100.0] * 60,
        [100.0] * 60
        + [math.nan]
        + [100.0] * 60,
        [100.0] * 60
        + [math.inf]
        + [100.0] * 60,
        [100.0] * 60
        + ["bad"]
        + [100.0] * 60,
        [100.0] * 60
        + [True]
        + [100.0] * 60,
    ],
)
def test_invalid_prices_are_rejected(
    bad_prices: list[object],
) -> None:
    with pytest.raises(
        VolatilityEngineError
    ):
        engine().classify(
            bad_prices
        )  # type: ignore[arg-type]


def test_insufficient_data_is_rejected() -> None:
    with pytest.raises(
        VolatilityEngineError
    ):
        engine().classify(
            [100.0] * 60,
            volatility_window=20,
            reference_window=40,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volatility_window": 1},
        {"reference_window": 1},
        {"ewma_decay": 0.0},
        {"ewma_decay": 1.0},
        {"high_ratio": 1.0},
        {"high_ratio": math.inf},
        {"extreme_ratio": 1.5},
        {"extreme_ratio": math.inf},
        {"low_ratio": 0.0},
        {"low_ratio": 1.0},
    ],
)
def test_invalid_parameters_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        VolatilityEngineError
    ):
        engine().classify(
            make_stable(),
            **kwargs,  # type: ignore[arg-type]
        )


def test_flat_market_has_zero_volatility() -> None:
    result = engine().classify(
        make_flat(),
        volatility_window=20,
        reference_window=40,
    )

    assert result.ewma_volatility == 0.0
    assert result.realized_volatility == 0.0
    assert result.reference_volatility == 0.0
    assert result.volatility_ratio == 1.0
    assert result.state == (
        VolatilityState.NORMAL
    )


def test_high_volatility_is_detected() -> None:
    result = engine().classify(
        make_volatility_expansion(),
        volatility_window=20,
        reference_window=40,
        high_ratio=1.50,
    )

    assert result.volatility_ratio >= 1.50

    assert result.state in {
        VolatilityState.HIGH,
        VolatilityState.EXTREME,
    }


def test_low_volatility_is_detected() -> None:
    result = engine().classify(
        make_volatility_compression(),
        volatility_window=20,
        reference_window=40,
        low_ratio=0.67,
    )

    assert result.volatility_ratio <= 0.67
    assert result.state == (
        VolatilityState.LOW
    )


def test_zero_reference_can_produce_infinite_ratio() -> None:
    prices = [100.0] * 41
    current = 100.0

    for i in range(20):
        direction = (
            1.0
            if i % 2 == 0
            else -1.0
        )

        current *= (
            1.0 + direction * 0.05
        )

        prices.append(current)

    result = engine().classify(
        prices,
        volatility_window=20,
        reference_window=20,
    )

    assert math.isinf(
        result.volatility_ratio
    )

    assert result.state == (
        VolatilityState.EXTREME
    )


def test_current_and_reference_windows_do_not_overlap() -> None:
    prices = make_volatility_expansion()

    baseline = engine().classify(
        prices,
        volatility_window=20,
        reference_window=40,
    )

    altered = prices.copy()

    start = len(altered) - 60
    end = len(altered) - 20

    for index in range(start, end):
        altered[index] *= 1.0000001

    changed = engine().classify(
        altered,
        volatility_window=20,
        reference_window=40,
    )

    assert changed.ewma_volatility == pytest.approx(
        baseline.ewma_volatility
    )


def test_same_input_is_deterministic() -> None:
    prices = make_volatility_expansion()

    first = engine().classify(prices)
    second = engine().classify(prices)

    assert first == second


def test_self_test_passes() -> None:
    assert engine()._self_test() is True
