from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.research_contract import (
    Decision,
    DecisionAction,
    ExecutionEvent,
    Forecast,
    Observation,
    Outcome,
    ResearchContractError,
    ResearchEpisode,
    validate_timeline,
)


T0 = 1_700_000_000_000
T1 = T0 + 60_000
T2 = T1 + 60_000
T3 = T2 + 60_000


def make_observation() -> Observation:
    return Observation(
        "btcusdt",
        "1m",
        T0,
        100_000.0,
    )


def make_forecast() -> Forecast:
    return Forecast(
        symbol="BTCUSDT",
        generated_at=T0,
        horizon_end=T2,
        expected_return=0.002,
        probability_positive=0.63,
        downside_probability=0.21,
        uncertainty=0.01,
        model_id="baseline-v1",
        feature_version="features-v1",
        dataset_version="dataset-v1",
    )


def make_decision() -> Decision:
    return Decision(
        symbol="BTCUSDT",
        decided_at=T1,
        forecast_generated_at=T0,
        action=DecisionAction.ENTER_LONG,
        expected_return=0.002,
        expected_cost=0.0004,
        risk_penalty=0.0005,
        expected_utility=0.0011,
    )


def make_execution() -> ExecutionEvent:
    return ExecutionEvent(
        symbol="BTCUSDT",
        decision_at=T1,
        executed_at=T1 + 1_000,
        action=DecisionAction.ENTER_LONG,
        reference_price=100_000.0,
        execution_price=100_010.0,
        quantity=0.001,
        total_cost=0.04,
    )


def make_outcome() -> Outcome:
    return Outcome(
        symbol="BTCUSDT",
        measured_at=T3,
        horizon_end=T2,
        realized_return=0.0017,
    )


def test_valid_full_episode_is_accepted() -> None:
    episode = ResearchEpisode(
        observation=make_observation(),
        forecast=make_forecast(),
        decision=make_decision(),
        execution=make_execution(),
        outcome=make_outcome(),
    )

    assert validate_timeline(episode) is True
    assert episode.is_executed is True
    assert episode.is_resolved is True

    assert [
        name
        for name, _
        in episode.timeline()
    ] == [
        "observation",
        "forecast",
        "decision",
        "execution",
        "outcome",
    ]


def test_no_trade_episode_can_exist_without_execution() -> None:
    decision = Decision(
        symbol="BTCUSDT",
        decided_at=T1,
        forecast_generated_at=T0,
        action=DecisionAction.NO_TRADE,
        expected_return=0.0001,
        expected_cost=0.0004,
        risk_penalty=0.001,
        expected_utility=-0.0013,
    )

    episode = ResearchEpisode(
        make_observation(),
        make_forecast(),
        decision,
    )

    assert episode.is_executed is False
    assert episode.is_resolved is False
    assert validate_timeline(episode) is True


def test_symbol_is_normalized() -> None:
    assert make_observation().symbol == "BTCUSDT"


def test_invalid_symbol_is_rejected() -> None:
    with pytest.raises(ResearchContractError):
        Observation(
            "BTC/USDT",
            "1m",
            T0,
            100_000.0,
        )


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ResearchContractError):
        Forecast(
            symbol="BTCUSDT",
            generated_at=T0,
            horizon_end=T2,
            expected_return=0.0,
            probability_positive=1.1,
            downside_probability=0.2,
            uncertainty=0.01,
            model_id="m",
            feature_version="f",
            dataset_version="d",
        )


def test_forecast_horizon_must_be_future() -> None:
    with pytest.raises(ResearchContractError):
        Forecast(
            symbol="BTCUSDT",
            generated_at=T1,
            horizon_end=T1,
            expected_return=0.0,
            probability_positive=0.5,
            downside_probability=0.5,
            uncertainty=0.01,
            model_id="m",
            feature_version="f",
            dataset_version="d",
        )


def test_forecast_cannot_precede_observation() -> None:
    forecast = Forecast(
        symbol="BTCUSDT",
        generated_at=T0 - 1,
        horizon_end=T2,
        expected_return=0.0,
        probability_positive=0.5,
        downside_probability=0.5,
        uncertainty=0.01,
        model_id="m",
        feature_version="f",
        dataset_version="d",
    )

    with pytest.raises(ResearchContractError):
        ResearchEpisode(
            make_observation(),
            forecast,
            make_decision(),
        )


def test_decision_cannot_use_future_forecast() -> None:
    with pytest.raises(ResearchContractError):
        Decision(
            symbol="BTCUSDT",
            decided_at=T0,
            forecast_generated_at=T1,
            action=DecisionAction.NO_TRADE,
            expected_return=0.0,
            expected_cost=0.0,
            risk_penalty=0.0,
            expected_utility=0.0,
        )


def test_execution_must_be_strictly_after_decision() -> None:
    with pytest.raises(ResearchContractError):
        ExecutionEvent(
            symbol="BTCUSDT",
            decision_at=T1,
            executed_at=T1,
            action=DecisionAction.ENTER_LONG,
            reference_price=100_000.0,
            execution_price=100_000.0,
            quantity=0.001,
            total_cost=0.0,
        )


def test_execution_must_reference_decision_timestamp() -> None:
    bad_execution = ExecutionEvent(
        symbol="BTCUSDT",
        decision_at=T1 + 10,
        executed_at=T2,
        action=DecisionAction.ENTER_LONG,
        reference_price=100_000.0,
        execution_price=100_001.0,
        quantity=0.001,
        total_cost=0.01,
    )

    with pytest.raises(ResearchContractError):
        ResearchEpisode(
            make_observation(),
            make_forecast(),
            make_decision(),
            execution=bad_execution,
        )


def test_outcome_cannot_be_measured_before_horizon() -> None:
    with pytest.raises(ResearchContractError):
        Outcome(
            symbol="BTCUSDT",
            measured_at=T1,
            horizon_end=T2,
            realized_return=0.0,
        )


def test_execution_action_must_match_decision() -> None:
    bad_execution = ExecutionEvent(
        symbol="BTCUSDT",
        decision_at=T1,
        executed_at=T1 + 1_000,
        action=DecisionAction.EXIT_LONG,
        reference_price=100_000.0,
        execution_price=100_001.0,
        quantity=0.001,
        total_cost=0.01,
    )

    with pytest.raises(ResearchContractError):
        ResearchEpisode(
            make_observation(),
            make_forecast(),
            make_decision(),
            execution=bad_execution,
        )


def test_execution_must_fit_inside_forecast_horizon() -> None:
    late_execution = ExecutionEvent(
        symbol="BTCUSDT",
        decision_at=T1,
        executed_at=T2 + 1_000,
        action=DecisionAction.ENTER_LONG,
        reference_price=100_000.0,
        execution_price=100_001.0,
        quantity=0.001,
        total_cost=0.01,
    )

    with pytest.raises(ResearchContractError):
        ResearchEpisode(
            make_observation(),
            make_forecast(),
            make_decision(),
            execution=late_execution,
        )


def test_no_trade_cannot_have_execution() -> None:
    no_trade = Decision(
        symbol="BTCUSDT",
        decided_at=T1,
        forecast_generated_at=T0,
        action=DecisionAction.NO_TRADE,
        expected_return=0.0,
        expected_cost=0.0,
        risk_penalty=0.0,
        expected_utility=0.0,
    )

    no_trade_execution = ExecutionEvent(
        symbol="BTCUSDT",
        decision_at=T1,
        executed_at=T1 + 1_000,
        action=DecisionAction.NO_TRADE,
        reference_price=100_000.0,
        execution_price=100_000.0,
        quantity=0.001,
        total_cost=0.0,
    )

    with pytest.raises(ResearchContractError):
        ResearchEpisode(
            make_observation(),
            make_forecast(),
            no_trade,
            execution=no_trade_execution,
        )


def test_outcome_horizon_must_match_forecast() -> None:
    bad_outcome = Outcome(
        symbol="BTCUSDT",
        measured_at=T3,
        horizon_end=T3,
        realized_return=0.0,
    )

    with pytest.raises(ResearchContractError):
        ResearchEpisode(
            make_observation(),
            make_forecast(),
            make_decision(),
            outcome=bad_outcome,
        )


def test_all_numeric_evidence_must_be_finite() -> None:
    with pytest.raises(ResearchContractError):
        Observation(
            "BTCUSDT",
            "1m",
            T0,
            float("nan"),
        )

    with pytest.raises(ResearchContractError):
        Forecast(
            symbol="BTCUSDT",
            generated_at=T0,
            horizon_end=T2,
            expected_return=float("inf"),
            probability_positive=0.5,
            downside_probability=0.5,
            uncertainty=0.01,
            model_id="m",
            feature_version="f",
            dataset_version="d",
        )


def test_frozen_contract_cannot_be_mutated() -> None:
    observation = make_observation()

    with pytest.raises(FrozenInstanceError):
        observation.close = 1.0  # type: ignore[misc]


def test_action_strings_are_normalized_to_enum() -> None:
    decision = Decision(
        symbol="BTCUSDT",
        decided_at=T1,
        forecast_generated_at=T0,
        action="NO_TRADE",  # type: ignore[arg-type]
        expected_return=0.0,
        expected_cost=0.0,
        risk_penalty=0.0,
        expected_utility=0.0,
    )

    assert decision.action is DecisionAction.NO_TRADE


def test_timeline_rejects_manual_out_of_order_state() -> None:
    episode = ResearchEpisode(
        make_observation(),
        make_forecast(),
        make_decision(),
    )

    object.__setattr__(
        episode.decision,
        "decided_at",
        T0 - 10,
    )

    with pytest.raises(ResearchContractError):
        validate_timeline(episode)
