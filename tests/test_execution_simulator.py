"""Phase 10 tests for the deterministic paper execution simulator."""

from __future__ import annotations

import math

import pytest

from core.execution_simulator import (
    ExecutionConfig,
    ExecutionError,
    ExecutionSide,
    ExecutionSimulator,
    ExecutionStatus,
)


def test_buy_execution_is_above_reference_price() -> None:
    simulator = ExecutionSimulator()

    fill = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    assert fill.status == ExecutionStatus.FILLED
    assert fill.fully_filled
    assert fill.filled_quantity == pytest.approx(1.0)
    assert fill.execution_price > 100.0
    assert fill.spread_cost > 0.0
    assert fill.slippage_cost > 0.0
    assert fill.impact_cost > 0.0
    assert fill.effective_cost_bps > 0.0


def test_sell_execution_is_below_reference_price() -> None:
    simulator = ExecutionSimulator()

    fill = simulator.market_order(
        ExecutionSide.SELL,
        quantity=1.0,
        reference_price=100.0,
        volatility=0.01,
        liquidity_notional=10_000.0,
    )

    assert fill.status == ExecutionStatus.FILLED
    assert fill.fully_filled
    assert fill.execution_price < 100.0
    assert fill.spread_cost > 0.0
    assert fill.slippage_cost > 0.0
    assert fill.impact_cost > 0.0
    assert fill.effective_cost_bps > 0.0


def test_zero_volatility_has_no_volatility_slippage_component() -> None:
    simulator = ExecutionSimulator(
        ExecutionConfig(
            half_spread_bps=1.0,
            base_slippage_bps=0.5,
            volatility_slippage_factor=0.05,
            impact_coefficient_bps=0.0,
        )
    )

    fill = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
        volatility=0.0,
    )

    assert fill.execution_price == pytest.approx(100.015)
    assert fill.impact_cost == pytest.approx(0.0)


def test_higher_volatility_increases_execution_cost() -> None:
    simulator = ExecutionSimulator(
        ExecutionConfig(
            impact_coefficient_bps=0.0,
        )
    )

    low = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
        volatility=0.005,
    )

    high = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
        volatility=0.02,
    )

    assert high.execution_price > low.execution_price
    assert high.effective_cost_bps > low.effective_cost_bps


def test_larger_participation_increases_market_impact() -> None:
    simulator = ExecutionSimulator()

    small = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
        liquidity_notional=10_000.0,
    )

    large = simulator.market_order(
        ExecutionSide.BUY,
        quantity=5.0,
        reference_price=100.0,
        liquidity_notional=10_000.0,
    )

    assert large.impact_bps > small.impact_bps
    assert large.impact_cost > small.impact_cost


def test_participation_limit_rejects_excessive_order() -> None:
    simulator = ExecutionSimulator(
        ExecutionConfig(
            max_participation_rate=0.10,
        )
    )

    fill = simulator.market_order(
        ExecutionSide.BUY,
        quantity=2.0,
        reference_price=100.0,
        liquidity_notional=1_000.0,
    )

    assert fill.status == ExecutionStatus.REJECTED
    assert fill.filled_quantity == 0.0
    assert fill.reason == "order-exceeds-participation-limit"


def test_missing_liquidity_can_be_required() -> None:
    simulator = ExecutionSimulator(
        ExecutionConfig(
            reject_when_liquidity_missing=True,
        )
    )

    fill = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
    )

    assert fill.status == ExecutionStatus.REJECTED
    assert fill.reason == "liquidity-notional-is-required"


def test_missing_liquidity_is_allowed_by_default() -> None:
    simulator = ExecutionSimulator()

    fill = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
    )

    assert fill.status == ExecutionStatus.FILLED
    assert fill.impact_bps == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("quantity", "price"),
    [
        (0.0, 100.0),
        (-1.0, 100.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (math.nan, 100.0),
        (1.0, math.inf),
    ],
)
def test_non_positive_or_non_finite_values_are_rejected(
    quantity: float,
    price: float,
) -> None:
    simulator = ExecutionSimulator()

    with pytest.raises(ExecutionError):
        simulator.market_order(
            ExecutionSide.BUY,
            quantity=quantity,
            reference_price=price,
        )


def test_invalid_side_is_rejected() -> None:
    simulator = ExecutionSimulator()

    with pytest.raises(ExecutionError):
        simulator.market_order(
            "SHORT",
            quantity=1.0,
            reference_price=100.0,
        )


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ExecutionError):
        ExecutionConfig(
            impact_exponent=0.0,
        )

    with pytest.raises(ExecutionError):
        ExecutionConfig(
            max_participation_rate=0.0,
        )


def test_json_representation_is_complete() -> None:
    simulator = ExecutionSimulator()

    fill = simulator.market_order(
        ExecutionSide.BUY,
        quantity=1.0,
        reference_price=100.0,
    )

    payload = fill.to_dict()

    assert payload["side"] == "BUY"
    assert payload["status"] == "FILLED"
    assert "impact_cost" in payload
    assert "effective_cost_bps" in payload


def test_execution_is_deterministic() -> None:
    simulator = ExecutionSimulator()

    kwargs = {
        "side": ExecutionSide.BUY,
        "quantity": 2.0,
        "reference_price": 125.0,
        "volatility": 0.012,
        "liquidity_notional": 20_000.0,
    }

    first = simulator.market_order(**kwargs)
    second = simulator.market_order(**kwargs)

    assert first == second
