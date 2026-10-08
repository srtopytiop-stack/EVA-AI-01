"""
EVA-AI-01 — Research Contract
==============================

V2 foundation layer.

This module defines the immutable evidence contract between:
    observation -> forecast -> decision -> execution -> outcome

It does not fetch market data, generate signals, place orders, train models,
or decide whether a strategy is profitable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ResearchContractError(ValueError):
    """Raised when a research-contract invariant is violated."""


class DecisionAction(str, Enum):
    """Research-level Spot decision vocabulary."""

    NO_TRADE = "NO_TRADE"
    ENTER_LONG = "ENTER_LONG"
    HOLD_LONG = "HOLD_LONG"
    EXIT_LONG = "EXIT_LONG"


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ResearchContractError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchContractError(f"{name} must be numeric") from exc
    if not isfinite(normalized) or normalized <= 0.0:
        raise ResearchContractError(f"{name} must be finite and > 0")
    return normalized


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ResearchContractError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchContractError(f"{name} must be numeric") from exc
    if not isfinite(normalized):
        raise ResearchContractError(f"{name} must be finite")
    return normalized


def _probability(value: object, name: str) -> float:
    normalized = _finite(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise ResearchContractError(f"{name} must be in [0, 1]")
    return normalized


def _timestamp(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchContractError(f"{name} must be an integer")
    if value <= 0:
        raise ResearchContractError(f"{name} must be > 0")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchContractError(f"{name} must be a non-empty string")
    return value.strip()


def _symbol(value: object) -> str:
    normalized = _text(value, "symbol").upper()
    if not normalized.isalnum():
        raise ResearchContractError("symbol must be alphanumeric")
    return normalized


@dataclass(frozen=True)
class Observation:
    """Point-in-time market observation available to the research system."""

    symbol: str
    interval: str
    observed_at: int
    close: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "interval", _text(self.interval, "interval"))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "observed_at"))
        object.__setattr__(self, "close", _positive_finite(self.close, "close"))


@dataclass(frozen=True)
class Forecast:
    """Probabilistic forecast produced from information available by generated_at.

    Probability fields are model outputs; calibration is a separate research
    question and is not implied by this contract.
    """

    symbol: str
    generated_at: int
    horizon_end: int
    expected_return: float
    probability_positive: float
    downside_probability: float
    uncertainty: float
    model_id: str
    feature_version: str
    dataset_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "generated_at", _timestamp(self.generated_at, "generated_at"))
        object.__setattr__(self, "horizon_end", _timestamp(self.horizon_end, "horizon_end"))
        if self.horizon_end <= self.generated_at:
            raise ResearchContractError("horizon_end must be strictly after generated_at")
        object.__setattr__(self, "expected_return", _finite(self.expected_return, "expected_return"))
        object.__setattr__(self, "probability_positive", _probability(self.probability_positive, "probability_positive"))
        object.__setattr__(self, "downside_probability", _probability(self.downside_probability, "downside_probability"))
        object.__setattr__(self, "uncertainty", _finite(self.uncertainty, "uncertainty"))
        if self.uncertainty < 0.0:
            raise ResearchContractError("uncertainty must be >= 0")
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "feature_version", _text(self.feature_version, "feature_version"))
        object.__setattr__(self, "dataset_version", _text(self.dataset_version, "dataset_version"))


@dataclass(frozen=True)
class Decision:
    """Decision derived from a forecast and the current research state."""

    symbol: str
    decided_at: int
    forecast_generated_at: int
    action: DecisionAction
    expected_return: float
    expected_cost: float
    risk_penalty: float
    expected_utility: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "decided_at", _timestamp(self.decided_at, "decided_at"))
        object.__setattr__(self, "forecast_generated_at", _timestamp(self.forecast_generated_at, "forecast_generated_at"))
        if self.forecast_generated_at > self.decided_at:
            raise ResearchContractError("forecast_generated_at cannot be after decided_at")
        if not isinstance(self.action, DecisionAction):
            try:
                object.__setattr__(self, "action", DecisionAction(str(self.action)))
            except ValueError as exc:
                raise ResearchContractError("action must be a valid DecisionAction") from exc
        for name in ("expected_return", "expected_cost", "risk_penalty", "expected_utility"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.expected_cost < 0.0:
            raise ResearchContractError("expected_cost must be >= 0")
        if self.risk_penalty < 0.0:
            raise ResearchContractError("risk_penalty must be >= 0")


@dataclass(frozen=True)
class ExecutionEvent:
    """Execution evidence; execution must occur strictly after the decision."""

    symbol: str
    decision_at: int
    executed_at: int
    action: DecisionAction
    reference_price: float
    execution_price: float
    quantity: float
    total_cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "decision_at", _timestamp(self.decision_at, "decision_at"))
        object.__setattr__(self, "executed_at", _timestamp(self.executed_at, "executed_at"))
        if self.executed_at <= self.decision_at:
            raise ResearchContractError("executed_at must be strictly after decision_at")
        if not isinstance(self.action, DecisionAction):
            try:
                object.__setattr__(self, "action", DecisionAction(str(self.action)))
            except ValueError as exc:
                raise ResearchContractError("action must be a valid DecisionAction") from exc
        object.__setattr__(self, "reference_price", _positive_finite(self.reference_price, "reference_price"))
        object.__setattr__(self, "execution_price", _positive_finite(self.execution_price, "execution_price"))
        object.__setattr__(self, "quantity", _positive_finite(self.quantity, "quantity"))
        object.__setattr__(self, "total_cost", _finite(self.total_cost, "total_cost"))
        if self.total_cost < 0.0:
            raise ResearchContractError("total_cost must be >= 0")


@dataclass(frozen=True)
class Outcome:
    """Observed realized outcome for a forecast/decision horizon."""

    symbol: str
    measured_at: int
    horizon_end: int
    realized_return: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(self, "measured_at", _timestamp(self.measured_at, "measured_at"))
        object.__setattr__(self, "horizon_end", _timestamp(self.horizon_end, "horizon_end"))
        if self.measured_at < self.horizon_end:
            raise ResearchContractError("measured_at must be >= horizon_end")
        object.__setattr__(self, "realized_return", _finite(self.realized_return, "realized_return"))


@dataclass(frozen=True)
class ResearchEpisode:
    """One auditable research episode; execution is optional for NO_TRADE."""

    observation: Observation
    forecast: Forecast
    decision: Decision
    execution: ExecutionEvent | None = None
    outcome: Outcome | None = None

    def __post_init__(self) -> None:
        symbol = self.observation.symbol
        if self.forecast.symbol != symbol:
            raise ResearchContractError("observation and forecast symbols must match")
        if self.decision.symbol != symbol:
            raise ResearchContractError("observation and decision symbols must match")
        if self.forecast.generated_at < self.observation.observed_at:
            raise ResearchContractError("forecast cannot be generated before its observation")
        if self.decision.forecast_generated_at != self.forecast.generated_at:
            raise ResearchContractError("decision must reference the forecast generation timestamp")
        if self.execution is not None:
            if self.execution.symbol != symbol:
                raise ResearchContractError("observation and execution symbols must match")
            if self.execution.decision_at != self.decision.decided_at:
                raise ResearchContractError("execution must reference the decision timestamp")
            if self.execution.action is not self.decision.action:
                raise ResearchContractError("execution action must match decision action")
            if self.decision.action is DecisionAction.NO_TRADE:
                raise ResearchContractError("NO_TRADE decisions cannot have an execution event")
            if self.execution.executed_at > self.forecast.horizon_end:
                raise ResearchContractError("execution must occur on or before the forecast horizon")
        if self.outcome is not None:
            if self.outcome.symbol != symbol:
                raise ResearchContractError("observation and outcome symbols must match")
            if self.outcome.horizon_end != self.forecast.horizon_end:
                raise ResearchContractError("outcome horizon_end must match forecast horizon_end")
            if self.outcome.measured_at < self.decision.decided_at:
                raise ResearchContractError("outcome cannot be measured before the decision")

    @property
    def is_executed(self) -> bool:
        return self.execution is not None

    @property
    def is_resolved(self) -> bool:
        return self.outcome is not None

    def timeline(self) -> tuple[tuple[str, int], ...]:
        values = [
            ("observation", self.observation.observed_at),
            ("forecast", self.forecast.generated_at),
            ("decision", self.decision.decided_at),
        ]
        if self.execution is not None:
            values.append(("execution", self.execution.executed_at))
        if self.outcome is not None:
            values.append(("outcome", self.outcome.measured_at))
        return tuple(values)


def validate_timeline(episode: ResearchEpisode) -> bool:
    """Validate non-decreasing research chronology."""
    timeline = episode.timeline()
    for (_, previous), (_, current) in zip(timeline, timeline[1:]):
        if current < previous:
            raise ResearchContractError("research timeline is not chronological")
    return True


__all__ = [
    "Decision",
    "DecisionAction",
    "ExecutionEvent",
    "Forecast",
    "Observation",
    "Outcome",
    "ResearchContractError",
    "ResearchEpisode",
    "validate_timeline",
]
