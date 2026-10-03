"""Tests for the EVA-AI-01 Phase 8 Risk Engine."""

from __future__ import annotations

import math

import pytest

from core.risk_engine import (
    RiskConfig,
    RiskDecision,
    RiskEngine,
    RiskEngineError,
    RiskResult,
    evaluate,
)


def engine() -> RiskEngine:
    return RiskEngine()


def test_module_and_facade_agree() -> None:
    module_result = evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=2.0,
        volatility_ratio=1.0,
    )

    facade_result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=2.0,
        volatility_ratio=1.0,
    )

    assert module_result == facade_result


def test_result_contains_expected_fields() -> None:
    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=2.0,
    )

    assert isinstance(result, RiskResult)
    assert isinstance(result.decision, RiskDecision)
    assert result.equity == 1_000.0
    assert result.entry_price == 100.0
    assert result.stop_distance == 2.0
    assert result.risk_budget > 0.0
    assert result.raw_quantity > 0.0
    assert result.adjusted_quantity > 0.0
    assert result.adjusted_notional > 0.0
    assert result.maximum_notional > 0.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.actual_risk_pct <= 0.01 + 1e-12


def test_default_risk_budget_and_quantity() -> None:
    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=5.0,
        volatility_ratio=1.0,
    )

    assert result.decision == RiskDecision.APPROVED
    assert result.risk_budget == pytest.approx(10.0)
    assert result.raw_quantity == pytest.approx(2.0)
    assert result.adjusted_quantity == pytest.approx(2.0)
    assert result.adjusted_notional == pytest.approx(200.0)
    assert result.risk_at_stop == pytest.approx(10.0)
    assert result.actual_risk_pct == pytest.approx(0.01)


def test_high_volatility_reduces_position_size() -> None:
    normal = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=5.0,
        volatility_ratio=1.0,
    )

    high_volatility = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=5.0,
        volatility_ratio=2.0,
    )

    assert high_volatility.decision == RiskDecision.REDUCED
    assert (
        high_volatility.adjusted_quantity
        < normal.adjusted_quantity
    )
    assert high_volatility.actual_risk_pct < normal.actual_risk_pct
    assert (
        "position-size-reduced-for-elevated-volatility"
        in high_volatility.reasons
    )


def test_exposure_cap_reduces_large_position() -> None:
    config = RiskConfig(
        risk_per_trade_pct=0.01,
        max_exposure_pct=0.05,
    )

    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=0.01,
        volatility_ratio=1.0,
        config=config,
    )

    assert result.decision == RiskDecision.REDUCED
    assert result.adjusted_notional == pytest.approx(50.0)
    assert result.adjusted_quantity == pytest.approx(0.5)
    assert (
        "position-size-capped-by-maximum-exposure"
        in result.reasons
    )


def test_spot_exposure_never_exceeds_equity() -> None:
    config = RiskConfig(max_exposure_pct=1.0)

    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=0.01,
        volatility_ratio=1.0,
        config=config,
    )

    assert result.adjusted_notional <= 1_000.0 + 1e-12


def test_very_high_volatility_hits_minimum_multiplier() -> None:
    config = RiskConfig(
        minimum_volatility_multiplier=0.25,
        maximum_volatility_multiplier=1.0,
    )

    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=2.0,
        volatility_ratio=100.0,
        config=config,
    )

    assert result.volatility_multiplier == pytest.approx(0.25)
    assert result.adjusted_quantity == pytest.approx(1.25)


def test_flat_volatility_ratio_is_neutral() -> None:
    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=2.0,
        volatility_ratio=1.0,
    )

    assert result.volatility_multiplier == pytest.approx(1.0)


def test_to_dict_is_json_friendly() -> None:
    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=2.0,
    )

    payload = result.to_dict()

    assert payload["decision"] == result.decision.value
    assert payload["quantity"] == result.quantity
    assert payload["notional"] == result.notional
    assert isinstance(payload["reasons"], list)


def test_invalid_equity_is_rejected() -> None:
    for equity in [0.0, -1.0, math.nan, math.inf]:
        with pytest.raises(RiskEngineError):
            engine().evaluate(
                equity=equity,
                entry_price=100.0,
                stop_distance=2.0,
            )


def test_invalid_entry_price_is_rejected() -> None:
    for entry_price in [0.0, -1.0, math.nan, math.inf]:
        with pytest.raises(RiskEngineError):
            engine().evaluate(
                equity=1_000.0,
                entry_price=entry_price,
                stop_distance=2.0,
            )


def test_invalid_stop_distance_is_rejected() -> None:
    for stop_distance in [0.0, -1.0, math.nan, math.inf, 100.0]:
        with pytest.raises(RiskEngineError):
            engine().evaluate(
                equity=1_000.0,
                entry_price=100.0,
                stop_distance=stop_distance,
            )


def test_invalid_volatility_ratio_is_rejected() -> None:
    for volatility_ratio in [0.0, -1.0, math.nan, math.inf]:
        with pytest.raises(RiskEngineError):
            engine().evaluate(
                equity=1_000.0,
                entry_price=100.0,
                stop_distance=2.0,
                volatility_ratio=volatility_ratio,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_per_trade_pct": 0.0},
        {"risk_per_trade_pct": 1.1},
        {"max_exposure_pct": 0.0},
        {"max_exposure_pct": 1.1},
        {"target_volatility_ratio": 0.0},
        {"minimum_volatility_multiplier": 0.0},
        {"minimum_volatility_multiplier": 1.1},
        {"maximum_volatility_multiplier": 0.0},
        {"maximum_volatility_multiplier": 1.1},
        {"maximum_stop_distance_pct": 0.0},
        {"maximum_stop_distance_pct": 1.0},
        {
            "minimum_volatility_multiplier": 0.9,
            "maximum_volatility_multiplier": 0.8,
        },
    ],
)
def test_invalid_config_is_rejected(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(RiskEngineError):
        RiskConfig(**kwargs)


def test_excessive_stop_distance_is_rejected() -> None:
    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=25.0,
    )

    assert result.decision == RiskDecision.REJECTED
    assert result.adjusted_quantity == 0.0
    assert result.adjusted_notional == 0.0
    assert (
        "stop-distance-exceeds-configured-risk-limit"
        in result.reasons
    )



def test_risk_engine_self_test_passes() -> None:
    assert engine()._self_test() is True


def test_output_is_deterministic() -> None:
    kwargs = {
        "equity": 1_000.0,
        "entry_price": 100.0,
        "stop_distance": 2.0,
        "volatility_ratio": 1.75,
    }

    first = engine().evaluate(**kwargs)
    second = engine().evaluate(**kwargs)

    assert first == second


def test_engine_does_not_contain_execution_fields() -> None:
    result = engine().evaluate(
        equity=1_000.0,
        entry_price=100.0,
        stop_distance=2.0,
    )

    assert not hasattr(result, "order_id")
    assert not hasattr(result, "exchange")
    assert not hasattr(result, "api_key")
    assert not hasattr(result, "leverage")
